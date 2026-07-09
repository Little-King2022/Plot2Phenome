# Copyright (c) OpenMMLab. All rights reserved.
"""Plot2Phenome instance segmentation head based on Mask2Former."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList
from .mask2former_head import Mask2FormerHead


@MODELS.register_module()
class Plot2PhenomeInstanceHead(Mask2FormerHead):
    """Mask2Former head adapted for true instance segmentation.

    The key change from the parent ``Mask2FormerHead`` is in
    ``_seg_data_to_instance_data``: instead of extracting unique class IDs
    from a semantic map (which merges all plot pixels into one "instance"),
    this head reads pre-computed per-instance binary masks from the data
    sample.  Each query in the Mask2Former decoder can then match to a
    distinct plot instance via Hungarian matching.

    Args:
        score_threshold: Minimum class score to keep an instance at inference.
        mask_threshold: Sigmoid threshold to binarise predicted masks.
        max_instances: Maximum number of instances to output per image.
    """

    def __init__(self,
                 num_classes: int,
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 score_threshold: float = 0.5,
                 mask_threshold: float = 0.5,
                 max_instances: int = 30,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            align_corners=align_corners,
            ignore_index=ignore_index,
            **kwargs)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.max_instances = max_instances

    # ------------------------------------------------------------------
    # Training: override GT conversion to use real instance masks
    # ------------------------------------------------------------------

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Convert data samples to instance data for Hungarian matching.

        If ``gt_instance_masks`` is present on the data sample, use the
        pre-computed per-instance masks directly.  Otherwise fall back to the
        parent's semantic-to-instance conversion for backward compatibility.
        """
        batch_img_metas: List[dict] = []
        batch_gt_instances: List[InstanceData] = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

            if (hasattr(data_sample, 'gt_instance_masks')
                    and hasattr(data_sample, 'gt_instance_labels')):
                gt_masks = data_sample.gt_instance_masks  # (N, H, W)
                gt_labels = data_sample.gt_instance_labels  # (N,)
                # Ensure tensors
                if not isinstance(gt_masks, Tensor):
                    gt_masks = torch.as_tensor(gt_masks)
                if not isinstance(gt_labels, Tensor):
                    gt_labels = torch.as_tensor(gt_labels)
                instance_data = InstanceData(
                    labels=gt_labels, masks=gt_masks.long())
            else:
                # Fallback: parent semantic → instance conversion
                gt_sem_seg = data_sample.gt_sem_seg.data
                classes = torch.unique(gt_sem_seg, sorted=False)
                gt_labels = classes[classes != self.ignore_index]
                masks = []
                for class_id in gt_labels:
                    masks.append(gt_sem_seg == class_id)
                if len(masks) == 0:
                    gt_masks = torch.zeros(
                        (0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]),
                        device=gt_sem_seg.device).long()
                else:
                    gt_masks = torch.stack(masks).squeeze(1).long()
                instance_data = InstanceData(
                    labels=gt_labels, masks=gt_masks)

            batch_gt_instances.append(instance_data)

        return batch_gt_instances, batch_img_metas

    # ------------------------------------------------------------------
    # Inference: instance-level prediction extraction
    # ------------------------------------------------------------------

    def predict_instances(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        test_cfg: ConfigType,
    ) -> List[InstanceData]:
        """Return per-instance masks, scores and labels.

        Unlike ``predict()`` which returns aggregated semantic logits, this
        method preserves individual query predictions as separate instances.
        """
        batch_data_samples = [
            SegDataSample(metainfo=m) for m in batch_img_metas
        ]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # Use last decoder layer predictions
        mask_cls = all_cls_scores[-1]    # (B, Q, num_classes + 1)
        mask_pred = all_mask_preds[-1]   # (B, Q, h, w)

        # Upsample to input resolution
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        mask_pred = F.interpolate(
            mask_pred, size=size, mode='bilinear', align_corners=False)

        results: List[InstanceData] = []
        for b in range(mask_cls.shape[0]):
            cls_score = F.softmax(mask_cls[b], dim=-1)  # (Q, C+1)
            pred_masks = mask_pred[b].sigmoid()          # (Q, H, W)

            # Score = probability of being class 1 (plot)
            # For multi-class, take max over foreground classes
            if self.num_classes == 2:
                scores = cls_score[:, 1]  # plot class
            else:
                scores, _ = cls_score[:, :self.num_classes].max(dim=-1)

            # Filter by score threshold
            keep = scores > self.score_threshold
            scores = scores[keep]
            masks = pred_masks[keep]  # (K, H, W)

            if len(scores) == 0:
                results.append(InstanceData(
                    masks=torch.zeros(
                        (0,) + size, device=mask_pred.device,
                        dtype=torch.bool),
                    scores=torch.zeros(0, device=mask_pred.device),
                    labels=torch.zeros(
                        0, device=mask_pred.device, dtype=torch.long)))
                continue

            # Binarise masks
            binary_masks = masks > self.mask_threshold

            # Remove empty masks
            mask_areas = binary_masks.sum(dim=(1, 2))
            valid = mask_areas > 0
            scores = scores[valid]
            binary_masks = binary_masks[valid]

            # Sort by score, truncate
            sorted_idx = scores.argsort(descending=True)
            n_keep = min(len(sorted_idx), self.max_instances)
            sorted_idx = sorted_idx[:n_keep]
            scores = scores[sorted_idx]
            binary_masks = binary_masks[sorted_idx]

            labels = torch.ones(
                len(scores), device=mask_pred.device, dtype=torch.long)

            results.append(InstanceData(
                masks=binary_masks,
                scores=scores,
                labels=labels))

        return results
