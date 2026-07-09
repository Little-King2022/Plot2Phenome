# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-scale spatial-channel attention module for Plot2Phenome."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class MSCAM(BaseModule):
    """Per-scale multi-scale spatial-channel attention block.

    The block keeps the input shape unchanged and applies residual attention
    before features are consumed by Mask2Former pixel decoder.
    """

    def __init__(self,
                 channels: int,
                 reduction: int = 4,
                 norm_cfg: Optional[dict] = dict(
                     type='BN', requires_grad=True),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 shape_check: bool = False,
                 gated_residual: bool = False,
                 predict_foreground: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.shape_check = shape_check
        self.gated_residual = gated_residual
        self.predict_foreground = predict_foreground
        hidden_channels = max(channels // reduction, 1)

        self.branch3 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.branch5 = ConvModule(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.branch_dilated = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mix_proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True), nn.Sigmoid())
        self.out_proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.out_act = nn.ReLU(inplace=True)

        # Learnable scalar gate on the refinement path (init "open" = 1.0).
        # Keeps the full model identical at init while exposing a measurable
        # contribution strength for the refinement.
        if gated_residual:
            self.res_gamma = nn.Parameter(torch.ones(()))
        # Optional foreground deep-supervision head on the multi-scale branch,
        # forcing MSCAM to encode plot regions instead of acting as identity.
        if predict_foreground:
            self.fg_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Refine one scale feature of shape ``[B, C, H, W]``.

        When ``predict_foreground`` is enabled the return is
        ``(refined, fg_logit)``; otherwise just ``refined``.
        """
        multi_scale = (self.branch3(x) + self.branch5(x) +
                       self.branch_dilated(x)) / 3.0
        multi_scale = self.mix_proj(multi_scale)

        gap = multi_scale.mean(dim=(2, 3), keepdim=True)
        gmp = multi_scale.amax(dim=(2, 3), keepdim=True)
        channel_weight = self.channel_mlp(gap + gmp)

        spatial_avg = multi_scale.mean(dim=1, keepdim=True)
        spatial_max = multi_scale.amax(dim=1, keepdim=True)
        spatial_weight = self.spatial_attn(
            torch.cat([spatial_avg, spatial_max], dim=1))

        attn_feat = multi_scale * channel_weight * spatial_weight
        attn_feat = self.out_proj(attn_feat)
        if self.gated_residual:
            refined = self.out_act(x + self.res_gamma * attn_feat)
        else:
            refined = self.out_act(x + attn_feat)

        if self.shape_check:
            assert refined.shape == x.shape, (
                'MSCAM output shape mismatch: '
                f'{refined.shape} vs {x.shape}.')

        if self.predict_foreground:
            fg_logit = self.fg_head(multi_scale)
            return refined, fg_logit
        return refined
