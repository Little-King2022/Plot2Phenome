# Copyright (c) OpenMMLab. All rights reserved.
"""Boundary Attention Module for Plot2Phenome instance segmentation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS


class _GradientConv(nn.Module):
    """Depthwise 3x3 conv initialised with Sobel-like filters for gradient
    detection.  Learns to adapt during training.

    .. deprecated::
        Kept for backward compatibility with existing checkpoints.
        New code should use :class:`_MultiDirGradientConv`.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False)
        self._init_sobel(channels)

    def _init_sobel(self, channels: int) -> None:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        with torch.no_grad():
            for i in range(channels):
                if i % 2 == 0:
                    self.dw.weight[i, 0] = sobel_x
                else:
                    self.dw.weight[i, 0] = sobel_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class _MultiDirGradientConv(nn.Module):
    """Depthwise 3x3 conv initialised with four-direction gradient kernels.

    Covers 0 deg (vertical edges), 90 deg (horizontal edges), 45 deg and
    135 deg diagonal edges.  All kernels are learnable and adapt during
    training; the initialisation merely provides orientation-aware starting
    points that accelerate convergence for boundary detection.
    """

    # Four canonical 3x3 gradient kernels.
    _KERNELS = {
        0: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],       # Sobel-X (0 deg)
        90: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],       # Sobel-Y (90 deg)
        45: [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],       # 45 deg diagonal
        135: [[0, -1, -2], [1, 0, -1], [2, 1, 0]],      # 135 deg diagonal
    }

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False)
        self._init_multi_dir(channels)

    def _init_multi_dir(self, channels: int) -> None:
        directions = list(self._KERNELS.values())
        n_dir = len(directions)
        with torch.no_grad():
            for i in range(channels):
                kernel = directions[i % n_dir]
                self.dw.weight[i, 0] = torch.tensor(
                    kernel, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class _StructureTensorGradient(nn.Module):
    """Rotation-invariant structure-tensor boundary detector (BAMv3 core).

    For each input channel, computes a per-pixel boundary response from the
    local structure tensor of the feature map.  Unlike Sobel-based gradient
    detectors, this module is **provably rotation-invariant**: the
    eigenvalues of the structure tensor are invariant under SO(2) rotations
    of the input, so the output is identical regardless of the input's
    global orientation.

    This matches the breeding-plot data prior, where boundary orientations
    are uniformly distributed in ``[0°, 180°)`` due to (a) real field tilt
    of 10–30° and (b) ±30° offline rotation augmentation in the dataset
    builder.  Any directional initialisation (Sobel-X/Y or 4-direction) is
    therefore an arbitrary prior; the structure tensor avoids that prior
    entirely while still encoding rich edge information.

    The boundary response combines three rotation-invariant descriptors:

    - **Energy**       :math:`E = \\sqrt{\\lambda_1 + \\lambda_2}` —
      total local gradient energy (== local mean of :math:`|\\nabla f|`).
    - **Anisotropy**   :math:`A = \\sqrt{\\lambda_1 - \\lambda_2}` —
      energy in the dominant direction; high for clean line edges.
    - **Coherence**    :math:`C = (\\lambda_1 - \\lambda_2)/(\\lambda_1 + \\lambda_2)` —
      normalised line-likeness in [0, 1]; ~1 for ideal edges, ~0 for
      isotropic textures or junction-style noise.

    These three are combined per-channel via three learnable scalars
    :math:`(w_E, w_A, w_C)`, giving only ``3*C`` parameters per scale —
    fewer than a learnable depthwise 3x3 conv (``9*C``).

    Args:
        channels: Number of input channels (depthwise).
        smooth_kernel: Spatial integration kernel size for the structure
            tensor.  Default 3.  Must be odd.
        eps: Numerical safety added inside sqrt and division.
        init_weights: Initial ``(w_E, w_A, w_C)``.  Default
            ``(1.0, 0.0, 1.0)`` — start from energy + coherence and let
            anisotropy join in if useful.
    """

    # Standard normalised Sobel kernels (factor 1/8 keeps responses in
    # roughly the same range as the input).  Stored as plain tensors and
    # registered as buffers in __init__ so they move with .to(device) but
    # are not part of self.parameters().
    _SOBEL_X = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]]) / 8.0
    _SOBEL_Y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]]) / 8.0

    def __init__(self,
                 channels: int,
                 smooth_kernel: int = 3,
                 eps: float = 1e-6,
                 init_weights: Tuple[float, float, float] = (
                     1.0, 0.0, 1.0)) -> None:
        super().__init__()
        if smooth_kernel % 2 == 0:
            raise ValueError(
                f'smooth_kernel must be odd, got {smooth_kernel}')
        self.channels = channels
        self.smooth_kernel = smooth_kernel
        self.eps = eps

        # Fixed depthwise Sobel kernels — registered as buffers, not
        # parameters, so they are NOT updated during training.
        sx = self._SOBEL_X.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        sy = self._SOBEL_Y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)

        # Per-channel mixing weights for (Energy, Anisotropy, Coherence).
        init = torch.tensor(init_weights, dtype=torch.float32)
        self.mix_weights = nn.Parameter(init.view(1, 3).repeat(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise gradients (fixed Sobel)
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)

        # Outer products
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy

        # Local spatial integration of the gradient outer-products.  This
        # turns per-pixel rank-1 outer products into a smoothed structure
        # tensor that captures the dominant local orientation.
        k = self.smooth_kernel
        Jxx = F.avg_pool2d(gxx, k, stride=1, padding=k // 2)
        Jyy = F.avg_pool2d(gyy, k, stride=1, padding=k // 2)
        Jxy = F.avg_pool2d(gxy, k, stride=1, padding=k // 2)

        # Closed-form eigenvalues of the 2x2 SPD matrix
        # J = [[Jxx, Jxy], [Jxy, Jyy]]
        # lambda_{1,2} = trace/2 ± sqrt((trace/2)^2 - det)
        # discriminant = (Jxx - Jyy)^2 / 4 + Jxy^2 ≥ 0  (always)
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        half_trace = 0.5 * trace
        discr = torch.sqrt(
            torch.clamp(half_trace * half_trace - det, min=self.eps))
        lambda1 = half_trace + discr
        lambda2 = half_trace - discr

        # Numerical clamps — both eigenvalues are theoretically ≥ 0 but
        # fp rounding can produce small negatives.
        lambda1 = torch.clamp(lambda1, min=0.0)
        lambda2 = torch.clamp(lambda2, min=0.0)

        # Three rotation-invariant boundary descriptors
        energy = torch.sqrt(lambda1 + lambda2 + self.eps)
        anisotropy = torch.sqrt(lambda1 - lambda2 + self.eps)
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + self.eps)

        # Per-channel mix.  mix_weights: (C, 3)
        we = self.mix_weights[:, 0].view(1, -1, 1, 1)
        wa = self.mix_weights[:, 1].view(1, -1, 1, 1)
        wc = self.mix_weights[:, 2].view(1, -1, 1, 1)
        return we * energy + wa * anisotropy + wc * coherence


class _PerScaleBoundaryBlock(BaseModule):
    """Single-scale boundary evidence extraction and feature modulation.

    Args:
        in_channels: Channel count of the input feature map.
        feat_channels: Internal projection channels for boundary evidence.
        norm_cfg: Normalization config.
        act_cfg: Activation config.
        use_modality_grads: If True, extract gradients from separate RGB and
            height features (dual-modal mode).
        use_multi_dir: If True, use four-direction gradient kernels (0, 45,
            90, 135 deg) instead of two-direction Sobel (x, y).  Legacy
            BAMv2 flag — kept for backward compatibility.  Ignored if
            ``gradient_type`` is set.
        gradient_type: Selects the gradient operator.  One of
            ``'sobel2'`` (legacy 2-direction Sobel, BAMv1),
            ``'sobel4'`` (BAMv2 4-direction Sobel),
            ``'structure_tensor'`` (BAMv3 rotation-invariant).  If None,
            falls back to ``use_multi_dir`` for backward compatibility.
        smooth_kernel: Spatial integration kernel for the structure
            tensor (only used when ``gradient_type='structure_tensor'``).
            Must be odd.  Default 3.
        st_init_weights: Initial ``(w_E, w_A, w_C)`` mixing weights for
            structure tensor mode.  Default ``(1.0, 0.0, 1.0)``.
        alpha_init: Initial value for the learnable modulation strength.
        per_channel_alpha: If True, use a per-channel alpha vector instead
            of a single scalar. This allows different channels to modulate
            boundary evidence at different strengths.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 norm_cfg: Dict,
                 act_cfg: Dict,
                 use_modality_grads: bool = True,
                 use_multi_dir: bool = False,
                 gradient_type: Optional[str] = None,
                 smooth_kernel: int = 3,
                 st_init_weights: Tuple[float, float, float] = (
                     1.0, 0.0, 1.0),
                 alpha_init: float = 0.1,
                 per_channel_alpha: bool = False) -> None:
        super().__init__()
        self.use_modality_grads = use_modality_grads

        # Resolve gradient operator selection.
        # gradient_type takes precedence; use_multi_dir is the legacy flag.
        if gradient_type is None:
            gradient_type = 'sobel4' if use_multi_dir else 'sobel2'
        self.gradient_type = gradient_type

        if gradient_type == 'sobel2':
            def _make_grad(c: int) -> nn.Module:
                return _GradientConv(c)
        elif gradient_type == 'sobel4':
            def _make_grad(c: int) -> nn.Module:
                return _MultiDirGradientConv(c)
        elif gradient_type == 'structure_tensor':
            def _make_grad(c: int) -> nn.Module:
                return _StructureTensorGradient(
                    c, smooth_kernel=smooth_kernel,
                    init_weights=st_init_weights)
        else:
            raise ValueError(
                f"Unknown gradient_type: {gradient_type!r}. "
                "Expected one of: 'sobel2', 'sobel4', 'structure_tensor'.")

        if use_modality_grads:
            self.rgb_grad = _make_grad(in_channels)
            self.height_grad = _make_grad(in_channels)
            evidence_in = in_channels * 3  # rgb_grad + height_grad + fused
        else:
            self.fused_grad = _make_grad(in_channels)
            evidence_in = in_channels * 2  # fused_grad + fused

        # Boundary evidence aggregation
        self.evidence_reduce = ConvModule(
            evidence_in, feat_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.evidence_refine = ConvModule(
            feat_channels, feat_channels, kernel_size=3, padding=1,
            groups=feat_channels, norm_cfg=norm_cfg, act_cfg=None)

        # Spatial attention
        self.attn_conv = nn.Conv2d(feat_channels, 1, kernel_size=1)

        # Learnable modulation strength
        if per_channel_alpha:
            self.alpha = nn.Parameter(
                torch.full((1, in_channels, 1, 1), alpha_init))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Project evidence to common feat_channels for FPN head
        self.proj = ConvModule(
            feat_channels, feat_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg) \
            if feat_channels != in_channels else nn.Identity()

    def forward(
        self,
        fused: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (modulated_feat, evidence_feat)."""
        if self.use_modality_grads and rgb is not None and height is not None:
            rg = self.rgb_grad(rgb)
            hg = self.height_grad(height)
            cat = torch.cat([rg, hg, fused], dim=1)
        else:
            fg = self.fused_grad(fused) if hasattr(self, 'fused_grad') \
                else self.rgb_grad(fused)
            cat = torch.cat([fg, fused], dim=1)

        evidence = self.evidence_reduce(cat)
        evidence = self.evidence_refine(evidence) + evidence  # residual

        attn = torch.sigmoid(self.attn_conv(evidence))  # (B, 1, H, W)
        modulated = fused * (1.0 + self.alpha * attn)

        return modulated, evidence


@MODELS.register_module()
class BoundaryAttentionModule(BaseModule):
    """Multi-scale boundary attention that modulates backbone features to
    enhance instance boundary responses.

    Placed between backbone output and pixel decoder in the segmentor.
    Exploits RGB texture gradients and DSM height gradients at each feature
    scale to generate boundary-aware attention maps.

    Args:
        in_channels_list: Per-scale channel counts, e.g. [256, 512, 1024, 2048].
        feat_channels: Internal projection channels for boundary evidence.
        norm_cfg: Normalisation config for internal convolutions.
        act_cfg: Activation config for internal convolutions.
        use_multi_dir: Use four-direction gradient kernels instead of
            two-direction Sobel.  Legacy BAMv2 flag.  Ignored if
            ``gradient_type`` is set.  Default: False.
        gradient_type: Selects gradient operator across all scales.  One
            of ``'sobel2'`` (BAMv1), ``'sobel4'`` (BAMv2),
            ``'structure_tensor'`` (BAMv3).  None → derived from
            ``use_multi_dir``.  Default: None.
        smooth_kernel: Spatial integration kernel for structure tensor
            (only used when ``gradient_type='structure_tensor'``).
            Default 3.
        st_init_weights: Initial ``(w_E, w_A, w_C)`` mixing weights for
            structure tensor mode.  Default ``(1.0, 0.0, 1.0)``.
        per_channel_alpha: Use per-channel modulation strength instead of
            a single scalar.  Default: False (backward compatible).
        scale_alpha_decay: When > 0, initialise alpha at coarser scales
            with smaller values: ``alpha_init * decay^scale_idx``.
            Default: 0 (disabled, uniform init).
    """

    def __init__(self,
                 in_channels_list: Sequence[int],
                 feat_channels: int = 256,
                 norm_cfg: Dict = dict(type='BN', requires_grad=True),
                 act_cfg: Dict = dict(type='ReLU'),
                 use_multi_dir: bool = False,
                 gradient_type: Optional[str] = None,
                 smooth_kernel: int = 3,
                 st_init_weights: Tuple[float, float, float] = (
                     1.0, 0.0, 1.0),
                 per_channel_alpha: bool = False,
                 scale_alpha_decay: float = 0,
                 deep_supervision: bool = False,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_scales = len(in_channels_list)
        self.feat_channels = feat_channels
        self.deep_supervision = deep_supervision

        self.blocks = nn.ModuleList()
        base_alpha = 0.1
        for scale_idx, ch in enumerate(in_channels_list):
            if scale_alpha_decay > 0:
                alpha_init = base_alpha * (scale_alpha_decay ** scale_idx)
            else:
                alpha_init = base_alpha
            self.blocks.append(
                _PerScaleBoundaryBlock(
                    in_channels=ch,
                    feat_channels=feat_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    use_modality_grads=True,
                    use_multi_dir=use_multi_dir,
                    gradient_type=gradient_type,
                    smooth_kernel=smooth_kernel,
                    st_init_weights=st_init_weights,
                    alpha_init=alpha_init,
                    per_channel_alpha=per_channel_alpha))

        # Final boundary prediction head
        self.boundary_head = nn.Conv2d(feat_channels, 1, kernel_size=1)

        # Optional per-scale deep-supervision heads.  Supervising each scale's
        # boundary evidence individually makes every scale block load-bearing
        # rather than letting the FPN sum hide collapsed scales.
        if deep_supervision:
            self.scale_boundary_heads = nn.ModuleList([
                nn.Conv2d(feat_channels, 1, kernel_size=1)
                for _ in range(self.num_scales)])

    def forward(
        self,
        fused_feats: Tuple[torch.Tensor, ...],
        rgb_feats: Optional[Tuple[torch.Tensor, ...]] = None,
        height_feats: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Union[Tuple[Tuple[torch.Tensor, ...], torch.Tensor],
               Tuple[Tuple[torch.Tensor, ...], torch.Tensor,
                     Tuple[torch.Tensor, ...]]]:
        """
        Args:
            fused_feats: Multi-scale features from backbone (C1..C4).
            rgb_feats: Per-scale RGB branch features (optional).
            height_feats: Per-scale height branch features (optional).

        Returns:
            modulated_feats: Same shapes as fused_feats, boundary-enhanced.
            boundary_pred: (B, 1, H_s0, W_s0) boundary logits at stride-4.
            aux_boundary_preds: Per-scale boundary logits (each at its own
                resolution) when ``deep_supervision`` is enabled. Omitted
                otherwise to preserve the legacy 2-tuple BAM interface.
        """
        modulated: List[torch.Tensor] = []
        aux_preds: Optional[List[torch.Tensor]] = (
            [] if self.deep_supervision else None)
        # Target size = finest scale
        target_h, target_w = fused_feats[0].shape[2:]

        fpn_sum = None
        for i in range(self.num_scales):
            rgb_i = rgb_feats[i] if rgb_feats is not None else None
            height_i = height_feats[i] if height_feats is not None else None
            mod_feat, evidence = self.blocks[i](fused_feats[i], rgb_i, height_i)
            modulated.append(mod_feat)

            if aux_preds is not None:
                # Per-scale boundary logit at this scale's own resolution.
                aux_preds.append(self.scale_boundary_heads[i](evidence))

            # FPN-style summation for boundary prediction
            lat = evidence  # already at feat_channels from evidence_reduce
            if lat.shape[2:] != (target_h, target_w):
                lat = F.interpolate(
                    lat, size=(target_h, target_w),
                    mode='bilinear', align_corners=False)
            fpn_sum = lat if fpn_sum is None else fpn_sum + lat

        boundary_pred = self.boundary_head(fpn_sum)  # (B, 1, H, W)
        aux_out = tuple(aux_preds) if aux_preds is not None else None
        if aux_out is None:
            return tuple(modulated), boundary_pred
        return tuple(modulated), boundary_pred, aux_out
