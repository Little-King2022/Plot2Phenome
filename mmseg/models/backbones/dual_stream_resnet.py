# Copyright (c) OpenMMLab. All rights reserved.
"""Dual-stream ResNet backbone for Plot2Phenome."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class DualStreamResNet(BaseModule):
    """Dual-stream RGB/height ResNet with stage-wise fusion.

    The input must be a stacked 4-channel tensor ``[B, 4, H, W]`` where the
    first three channels are RGB and the last channel is height.
    """

    def __init__(self,
                 rgb_backbone: Dict,
                 height_backbone: Dict,
                 fusion_stages: Sequence[int] = (0, 1, 2),
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 hbfm_cfg: Optional[Dict] = None,
                 mscam_cfg: Optional[Dict] = None,
                 cmgfa_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = dict(
                     type='BN', requires_grad=True),
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 shape_check: bool = False,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.height_backbone = MODELS.build(height_backbone)
        self.fusion_stages = tuple(sorted(fusion_stages))
        self.out_indices = out_indices
        self.shape_check = shape_check

        self.stage_channels = self._infer_stage_channels(self.rgb_backbone)
        height_stage_channels = self._infer_stage_channels(self.height_backbone)
        if self.stage_channels != height_stage_channels:
            raise ValueError('RGB and height backbones must expose identical '
                             'stage channels for Plot2Phenome fusion.')
        self.out_channels = [self.stage_channels[idx] for idx in out_indices]

        # Determine fusion mode: CMGFA (new) or HBFM+MSCAM (legacy).
        self.use_cmgfa = cmgfa_cfg is not None

        if self.use_cmgfa:
            self._build_cmgfa_modules(cmgfa_cfg, norm_cfg)
        else:
            self._build_hbfm_mscam_modules(hbfm_cfg, mscam_cfg, norm_cfg,
                                           act_cfg)

    # ------------------------------------------------------------------
    # Module builders
    # ------------------------------------------------------------------

    def _build_cmgfa_modules(self, cmgfa_cfg: Dict,
                             norm_cfg: Optional[Dict]) -> None:
        """Build CMGFA fusion modules for all stages.

        CMGFA replaces both HBFM and MSCAM — each stage gets a single
        cross-modal gated fusion attention module.  Re-injection is
        applied at ``fusion_stages`` only (typically stages 0-2).
        """
        cmgfa_cfg = copy.deepcopy(cmgfa_cfg)
        self.cmgfa_modules = nn.ModuleDict()
        self.rgb_reinjectors = nn.ModuleDict()
        self.height_reinjectors = nn.ModuleDict()

        for stage_idx, stage_ch in enumerate(self.stage_channels):
            stage_cfg = copy.deepcopy(cmgfa_cfg)
            stage_cfg.setdefault('type', 'CMGFA')
            stage_cfg.setdefault('channels', stage_ch)
            stage_cfg.setdefault('shape_check', self.shape_check)
            self.cmgfa_modules[str(stage_idx)] = MODELS.build(stage_cfg)

            if stage_idx in self.fusion_stages:
                self.rgb_reinjectors[str(stage_idx)] = ConvModule(
                    stage_ch, stage_ch, kernel_size=1,
                    norm_cfg=norm_cfg, act_cfg=None)
                self.height_reinjectors[str(stage_idx)] = ConvModule(
                    stage_ch, stage_ch, kernel_size=1,
                    norm_cfg=norm_cfg, act_cfg=None)

    def _build_hbfm_mscam_modules(self, hbfm_cfg: Optional[Dict],
                                   mscam_cfg: Optional[Dict],
                                   norm_cfg: Optional[Dict],
                                   act_cfg: Optional[Dict]) -> None:
        """Build legacy HBFM + MSCAM modules (backward compatible)."""
        self.hbfm_modules = nn.ModuleDict()
        self.rgb_reinjectors = nn.ModuleDict()
        self.height_reinjectors = nn.ModuleDict()
        hbfm_cfg = copy.deepcopy(hbfm_cfg or dict(type='HBFM'))
        for stage_idx in self.fusion_stages:
            stage_cfg = copy.deepcopy(hbfm_cfg)
            stage_cfg.setdefault('type', 'HBFM')
            stage_cfg.setdefault('channels', self.stage_channels[stage_idx])
            stage_cfg.setdefault('shape_check', self.shape_check)
            self.hbfm_modules[str(stage_idx)] = MODELS.build(stage_cfg)
            self.rgb_reinjectors[str(stage_idx)] = ConvModule(
                self.stage_channels[stage_idx],
                self.stage_channels[stage_idx],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.height_reinjectors[str(stage_idx)] = ConvModule(
                self.stage_channels[stage_idx],
                self.stage_channels[stage_idx],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None)

        c5_channels = self.stage_channels[-1]
        self.c5_fusion = nn.Sequential(
            ConvModule(
                c5_channels * 2,
                c5_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                c5_channels,
                c5_channels,
                kernel_size=3,
                padding=1,
                groups=c5_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        mscam_cfg = copy.deepcopy(mscam_cfg or dict(type='MSCAM'))
        self.mscam_modules = nn.ModuleList()
        for stage_channels in self.stage_channels:
            stage_cfg = copy.deepcopy(mscam_cfg)
            stage_cfg.setdefault('type', 'MSCAM')
            stage_cfg.setdefault('channels', stage_channels)
            stage_cfg.setdefault('shape_check', self.shape_check)
            self.mscam_modules.append(MODELS.build(stage_cfg))

    @staticmethod
    def _infer_stage_channels(backbone: nn.Module) -> List[int]:
        if not all(
                hasattr(backbone, attr)
                for attr in ('block', 'base_channels', 'num_stages', 'res_layers')):
            raise TypeError('DualStreamResNet expects ResNet-like backbones.')
        return [
            backbone.base_channels * (2**idx) * backbone.block.expansion
            for idx in range(backbone.num_stages)
        ]

    @staticmethod
    def _forward_stem(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if getattr(backbone, 'deep_stem', False):
            x = backbone.stem(x)
        else:
            x = backbone.conv1(x)
            x = backbone.norm1(x)
            x = backbone.relu(x)
        x = backbone.maxpool(x)
        return x

    @staticmethod
    def _forward_stage(backbone: nn.Module, stage_idx: int,
                       x: torch.Tensor) -> torch.Tensor:
        layer_name = backbone.res_layers[stage_idx]
        layer = getattr(backbone, layer_name)
        return layer(x)

    def forward(self, inputs: torch.Tensor) -> Dict[str, Tuple[torch.Tensor,
                                                               ...]]:
        """Forward function.

        Args:
            inputs (Tensor): Stacked input tensor of shape ``[B, 4, H, W]``.

        Returns:
            dict: Fused features and modality-specific stage features.
        """
        if inputs.dim() != 4 or inputs.size(1) < 4:
            raise AssertionError(
                'DualStreamResNet expects BCHW input with at least 4 channels.')

        rgb = inputs[:, :3, :, :]
        height = inputs[:, 3:4, :, :]

        rgb_x = self._forward_stem(self.rgb_backbone, rgb)
        height_x = self._forward_stem(self.height_backbone, height)

        rgb_feats: List[torch.Tensor] = []
        height_feats: List[torch.Tensor] = []
        fused_feats: List[torch.Tensor] = []
        hbfm_boundary_preds: List[torch.Tensor] = []
        mscam_fg_preds: List[torch.Tensor] = []

        for stage_idx in range(len(self.stage_channels)):
            rgb_stage = self._forward_stage(self.rgb_backbone, stage_idx, rgb_x)
            height_stage = self._forward_stage(self.height_backbone, stage_idx,
                                               height_x)
            rgb_feats.append(rgb_stage)
            height_feats.append(height_stage)

            if self.use_cmgfa:
                fused = self._forward_cmgfa_stage(
                    stage_idx, rgb_stage, height_stage)
            else:
                fused, hbfm_aux, mscam_aux = self._forward_hbfm_mscam_stage(
                    stage_idx, rgb_stage, height_stage)
                if hbfm_aux is not None:
                    hbfm_boundary_preds.append(hbfm_aux)
                if mscam_aux is not None:
                    mscam_fg_preds.append(mscam_aux)

            if stage_idx in self.fusion_stages:
                rgb_x = rgb_stage + self.rgb_reinjectors[
                    str(stage_idx)](fused)
                height_x = height_stage + self.height_reinjectors[
                    str(stage_idx)](fused)
            else:
                rgb_x = rgb_stage
                height_x = height_stage

            if self.shape_check:
                assert fused.shape[1] == self.stage_channels[stage_idx], (
                    'Unexpected fused channels at stage '
                    f'{stage_idx}: {fused.shape}.')

            fused_feats.append(fused)

        selected_features = tuple(fused_feats[idx] for idx in self.out_indices)
        outputs = dict(
            features=selected_features,
            rgb_feats=tuple(rgb_feats),
            height_feats=tuple(height_feats),
            fused_feats=tuple(fused_feats))
        if hbfm_boundary_preds:
            outputs['hbfm_boundary_preds'] = tuple(hbfm_boundary_preds)
        if mscam_fg_preds:
            outputs['mscam_fg_preds'] = tuple(mscam_fg_preds)
        return outputs

    # ------------------------------------------------------------------
    # Per-stage fusion helpers
    # ------------------------------------------------------------------

    def _forward_cmgfa_stage(self, stage_idx: int,
                             rgb_stage: torch.Tensor,
                             height_stage: torch.Tensor) -> torch.Tensor:
        """CMGFA mode: single module replaces both HBFM and MSCAM."""
        return self.cmgfa_modules[str(stage_idx)](rgb_stage, height_stage)

    def _forward_hbfm_mscam_stage(
        self, stage_idx: int, rgb_stage: torch.Tensor,
        height_stage: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Legacy mode: HBFM fusion + MSCAM refinement.

        Returns ``(fused, hbfm_boundary_logit, mscam_fg_logit)``.  The two
        auxiliary logits are None unless the corresponding module has deep
        supervision enabled (``predict_boundary`` / ``predict_foreground``).
        """
        hbfm_aux = None
        if stage_idx in self.fusion_stages:
            out = self.hbfm_modules[str(stage_idx)](rgb_stage, height_stage)
            if isinstance(out, tuple):
                fused, hbfm_aux = out
            else:
                fused = out
        else:
            fused = self.c5_fusion(
                torch.cat([rgb_stage, height_stage], dim=1))

        mscam_aux = None
        mscam_out = self.mscam_modules[stage_idx](fused)
        if isinstance(mscam_out, tuple):
            refined, mscam_aux = mscam_out
        else:
            refined = mscam_out
        return refined, hbfm_aux, mscam_aux
