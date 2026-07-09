# Copyright (c) OpenMMLab. All rights reserved.
"""Alignment loss for Plot2Phenome dual-modal features."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class AlignmentLoss(BaseModule):
    """Cosine alignment loss on pooled RGB and height features."""

    def __init__(self,
                 in_channels: int,
                 proj_channels: int = 256,
                 loss_weight: float = 0.02,
                 eps: float = 1e-6,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss_weight = loss_weight
        self.eps = eps
        self.rgb_proj = nn.Linear(in_channels, proj_channels)
        self.height_proj = nn.Linear(in_channels, proj_channels)
        self.rgb_norm = nn.LayerNorm(proj_channels)
        self.height_norm = nn.LayerNorm(proj_channels)

    @staticmethod
    def _pool_feature(feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 4:
            return feature.mean(dim=(2, 3))
        if feature.dim() == 2:
            return feature
        raise ValueError(
            f'AlignmentLoss expects 2D or 4D features, got {feature.shape}.')

    def forward(self, rgb_feat: torch.Tensor,
                height_feat: torch.Tensor) -> torch.Tensor:
        rgb_feat = self._pool_feature(rgb_feat)
        height_feat = self._pool_feature(height_feat)
        if rgb_feat.shape != height_feat.shape:
            raise AssertionError(
                'AlignmentLoss expects matching pooled feature shapes, got '
                f'{rgb_feat.shape} and {height_feat.shape}.')

        rgb_feat = self.rgb_norm(self.rgb_proj(rgb_feat))
        height_feat = self.height_norm(self.height_proj(height_feat))

        cosine = F.cosine_similarity(rgb_feat, height_feat, dim=1, eps=self.eps)
        loss = 1 - cosine.mean()
        return loss * self.loss_weight
