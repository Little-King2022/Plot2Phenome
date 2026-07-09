# Copyright (c) OpenMMLab. All rights reserved.
"""Hierarchical bi-modal fusion module used by Plot2Phenome."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class HBFM(BaseModule):
    """Approximate HBFM with a light-weight wavelet-style height branch.

    Args:
        channels (int): Input and output channels for both modalities.
        norm_cfg (dict, optional): Normalization config.
        act_cfg (dict, optional): Activation config.
        low_freq_kernel (int): Kernel size for low-frequency pooling.
        shape_check (bool): Whether to assert critical tensor shapes.
        residual_mode (str): How the additive residual shortcut is formed.
            ``'concat'`` (default) projects ``concat[rgb, height]`` — the
            legacy behaviour, which lets the module bypass the wavelet
            fusion branch (it can approximate plain concat fusion).
            ``'rgb_only'`` projects only the RGB feature, forcing height
            information to flow through the fusion branch (makes the module
            load-bearing).  ``'none'`` removes the shortcut entirely.
        predict_boundary (bool): If True, attach a 1x1 head on the
            high-frequency height branch and return an auxiliary boundary
            logit for deep supervision.  This forces the high-frequency
            branch to encode real plot boundaries instead of collapsing.
    """

    def __init__(self,
                 channels: int,
                 norm_cfg: Optional[dict] = dict(
                     type='BN', requires_grad=True),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 low_freq_kernel: int = 2,
                 shape_check: bool = False,
                 residual_mode: str = 'concat',
                 predict_boundary: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.shape_check = shape_check
        if residual_mode not in ('concat', 'rgb_only', 'none'):
            raise ValueError(
                "residual_mode must be one of 'concat', 'rgb_only', 'none', "
                f'got {residual_mode!r}.')
        self.residual_mode = residual_mode
        self.predict_boundary = predict_boundary

        self.rgb_branch = nn.Sequential(
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        self.low_pool = nn.AvgPool2d(
            kernel_size=low_freq_kernel,
            stride=low_freq_kernel,
            ceil_mode=False)
        self.height_low_proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.height_high_proj = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.fusion_reduce = ConvModule(
            channels * 3,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fusion_dw = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fusion_expand = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        if residual_mode == 'concat':
            self.residual_proj = ConvModule(
                channels * 2,
                channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None)
        elif residual_mode == 'rgb_only':
            self.residual_proj = ConvModule(
                channels,
                channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:  # 'none'
            self.residual_proj = None
        self.out_act = nn.ReLU(inplace=True)

        # Optional deep-supervision head on the high-frequency branch.
        if predict_boundary:
            self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self, rgb_feat: torch.Tensor, height_feat: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Fuse same-shape RGB and height features.

        Both inputs are expected to have shape ``[B, C, H, W]``.  When
        ``predict_boundary`` is enabled the return is
        ``(fused, boundary_logit)``; otherwise just ``fused``.
        """
        if self.shape_check:
            assert rgb_feat.shape == height_feat.shape, (
                'HBFM expects RGB and height features with identical shapes, '
                f'got {rgb_feat.shape} and {height_feat.shape}.')

        rgb_local = self.rgb_branch(rgb_feat)

        low = self.low_pool(height_feat)
        low = F.interpolate(
            low,
            size=height_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        low = self.height_low_proj(low)
        high = self.height_high_proj(height_feat - low)

        fusion_input = torch.cat([rgb_local, low, high], dim=1)
        fused = self.fusion_reduce(fusion_input)
        fused = self.fusion_dw(fused)
        fused = self.fusion_expand(fused)

        if self.residual_mode == 'concat':
            residual = self.residual_proj(
                torch.cat([rgb_feat, height_feat], dim=1))
            fused = fused + residual
        elif self.residual_mode == 'rgb_only':
            # RGB-only shortcut: height must flow through the fusion branch.
            fused = fused + self.residual_proj(rgb_feat)
        # 'none': no additive shortcut.
        fused = self.out_act(fused)

        if self.shape_check:
            assert fused.shape == rgb_feat.shape, (
                'HBFM output shape mismatch: '
                f'{fused.shape} vs {rgb_feat.shape}.')

        if self.predict_boundary:
            # Deep-supervision logit from the high-frequency height branch.
            boundary_logit = self.boundary_head(high)
            return fused, boundary_logit
        return fused
