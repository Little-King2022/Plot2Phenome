# Copyright (c) OpenMMLab. All rights reserved.
"""Cross-Modal Gated Fusion Attention module for Plot2Phenome.

Replaces the separate HBFM + MSCAM pipeline with a single unified module
that performs bidirectional cross-modal gating and difference-aware
multi-scale refinement.

Design motivation (breeding plot segmentation):
- DSM height features carry strong boundary cues (abrupt height changes
  between adjacent plots) that should spatially guide RGB feature attention.
- RGB texture features indicate which height channels are semantically
  meaningful versus noise (e.g., wind-induced canopy sway).
- The absolute difference between modality features highlights cross-modal
  disagreement, which concentrates at plot boundaries — especially during
  canopy closure when RGB contrast diminishes but height contrast persists.

References:
- CMX (Zhang et al., TITS 2023): Cross-Modal Feature Rectification Module
  for bidirectional feature calibration.
- MSFFNet (Zhi et al., 2025): Wavelet-based frequency fusion for RGB+DSM.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class CMGFA(BaseModule):
    """Cross-Modal Gated Fusion Attention.

    Fuses same-shape RGB and height features through three mechanisms:

    1. **Bidirectional Gating**: height features generate a spatial gate
       for RGB (where to attend), while RGB features generate a channel
       gate for height (what to attend to).
    2. **Gated Feature Fusion**: gated features are concatenated and
       projected to produce a fused representation.
    3. **Difference-Aware Multi-Scale Refinement**: the absolute difference
       between the two modalities highlights boundary regions and drives
       a lightweight multi-scale attention that sharpens the fused output.

    Args:
        channels (int): Input and output channel count for both modalities.
        reduction (int): Channel reduction ratio for SE-style MLP and
            difference attention. Default: 4.
        ms_kernels (tuple[int, ...]): Kernel sizes for multi-scale
            depthwise branches. Default: (3, 5).
        norm_cfg (dict | None): Normalization config for ConvModule.
        act_cfg (dict | None): Activation config for ConvModule.
        shape_check (bool): Whether to assert critical tensor shapes.
        init_cfg (dict | None): Initialization config.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        ms_kernels: tuple = (3, 5),
        norm_cfg: Optional[dict] = dict(type='BN', requires_grad=True),
        act_cfg: Optional[dict] = dict(type='ReLU'),
        shape_check: bool = False,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.shape_check = shape_check
        hidden = max(channels // reduction, 1)

        # ------------------------------------------------------------------
        # 1. Bidirectional cross-modal gating
        # ------------------------------------------------------------------
        # Height -> RGB spatial gate: depthwise conv extracts local height
        # structure (gradients, edges), then 1x1 projects to per-channel
        # spatial attention for RGB features.
        self.h2r_gate = nn.Sequential(
            ConvModule(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())

        # RGB -> Height channel gate: GAP squeezes spatial information,
        # then a lightweight MLP selects informative height channels
        # based on RGB semantic context.
        self.r2h_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid())

        # ------------------------------------------------------------------
        # 2. Gated feature fusion
        # ------------------------------------------------------------------
        self.fusion_proj = ConvModule(
            channels * 2, channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ------------------------------------------------------------------
        # 3. Difference-aware multi-scale refinement
        # ------------------------------------------------------------------
        self.ms_branches = nn.ModuleList()
        for k in ms_kernels:
            self.ms_branches.append(
                ConvModule(
                    channels, channels, kernel_size=k,
                    padding=k // 2, groups=channels,
                    norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.ms_mix = ConvModule(
            channels, channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Difference attention: |rgb - height| encodes cross-modal
        # disagreement → lightweight bottleneck → spatial weight map.
        self.diff_attn = nn.Sequential(
            ConvModule(
                channels, hidden, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

        self.refine_proj = ConvModule(
            channels, channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)

        # ------------------------------------------------------------------
        # 4. Residual connection from raw inputs
        # ------------------------------------------------------------------
        self.residual_proj = ConvModule(
            channels * 2, channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, rgb_feat: torch.Tensor,
                height_feat: torch.Tensor) -> torch.Tensor:
        """Fuse same-shape RGB and height features.

        Args:
            rgb_feat: ``[B, C, H, W]`` RGB branch features.
            height_feat: ``[B, C, H, W]`` height branch features.

        Returns:
            Tensor: Fused features of shape ``[B, C, H, W]``.
        """
        if self.shape_check:
            assert rgb_feat.shape == height_feat.shape, (
                'CMGFA expects identical shapes, got '
                f'{rgb_feat.shape} vs {height_feat.shape}.')

        # --- Step 1: Bidirectional gating ---
        # Height tells RGB *where* to attend (spatial gate).
        h2r = self.h2r_gate(height_feat)        # (B, C, H, W)
        rgb_gated = rgb_feat * h2r

        # RGB tells height *what* to attend to (channel gate).
        r2h = self.r2h_gate(rgb_feat)            # (B, C, 1, 1)
        height_gated = height_feat * r2h

        # --- Step 2: Fuse gated features ---
        fused = self.fusion_proj(
            torch.cat([rgb_gated, height_gated], dim=1))  # (B, C, H, W)

        # --- Step 3: Difference-aware multi-scale refinement ---
        ms_sum = sum(branch(fused) for branch in self.ms_branches)
        ms_out = self.ms_mix(ms_sum / len(self.ms_branches))

        diff = torch.abs(rgb_feat - height_feat)
        diff_weight = self.diff_attn(diff)       # (B, 1, H, W)
        refined = self.refine_proj(ms_out * diff_weight)

        # --- Step 4: Residual + output ---
        residual = self.residual_proj(
            torch.cat([rgb_feat, height_feat], dim=1))
        return self.out_act(fused + refined + residual)
