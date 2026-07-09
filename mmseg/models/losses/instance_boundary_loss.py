# Copyright (c) OpenMMLab. All rights reserved.
"""Boundary loss combining weighted BCE and Dice for instance boundary
supervision."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class InstanceBoundaryLoss(nn.Module):
    """Combined weighted-BCE + Dice loss for boundary prediction.

    The weighted BCE follows the pattern from ``BoundaryLoss`` in
    ``mmseg/models/losses/boundary_loss.py`` — positive and negative samples
    are reweighted by their inverse frequency to handle boundary sparsity.

    Args:
        bce_weight: Relative weight for the BCE term.
        dice_weight: Relative weight for the Dice term.
        loss_weight: Overall scaling factor applied after combining BCE + Dice.
        eps: Numerical stability constant for Dice denominator.
        loss_name: Name used in the loss dict.
    """

    def __init__(self,
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 loss_weight: float = 0.5,
                 eps: float = 1.0,
                 loss_name: str = 'loss_boundary') -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.loss_weight = loss_weight
        self.eps = eps
        self.loss_name_ = loss_name

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: (B, 1, H_p, W_p) boundary logits.
            target: (B, 1, H_t, W_t) binary boundary ground truth.

        Returns:
            Scalar loss tensor.
        """
        # Align spatial resolution
        if pred.shape[2:] != target.shape[2:]:
            target = F.interpolate(
                target.float(), size=pred.shape[2:], mode='nearest')
        target = target.float()

        # --- Weighted BCE ---
        flat_pred = pred.reshape(1, -1)
        flat_target = target.reshape(1, -1)

        pos_mask = (flat_target == 1)
        neg_mask = (flat_target == 0)
        pos_num = pos_mask.sum().clamp(min=1)
        neg_num = neg_mask.sum().clamp(min=1)
        total = pos_num + neg_num

        weight = torch.zeros_like(flat_pred)
        weight[pos_mask] = neg_num.float() / total.float()
        weight[neg_mask] = pos_num.float() / total.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            flat_pred, flat_target, weight, reduction='mean')

        # --- Dice loss ---
        pred_sigmoid = pred.sigmoid()
        pred_flat = pred_sigmoid.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = 1.0 - (2.0 * intersection + self.eps) / (
            pred_flat.sum() + target_flat.sum() + self.eps)

        loss = (self.bce_weight * bce_loss +
                self.dice_weight * dice) * self.loss_weight
        return loss

    @property
    def loss_name(self) -> str:
        return self.loss_name_
