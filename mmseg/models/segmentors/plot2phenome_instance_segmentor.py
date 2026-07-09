# Copyright (c) OpenMMLab. All rights reserved.
"""Plot2Phenome instance segmentor with boundary attention."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from .plot2phenome_mask2former import Plot2PhenomeMask2Former


@MODELS.register_module()
class Plot2PhenomeInstanceSegmentor(Plot2PhenomeMask2Former):
    """Instance segmentor extending Plot2PhenomeMask2Former with boundary
    attention and instance-level output.

    Adds:
        1. ``BoundaryAttentionModule`` applied between backbone and decode head
           to modulate features near instance boundaries.
        2. ``InstanceBoundaryLoss`` supervised by a boundary ground-truth map.
        3. Instance-level prediction via ``decode_head.predict_instances()``.
    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_align: OptConfigType = None,
                 alignment_stage: int = 2,
                 boundary_attention: OptConfigType = None,
                 loss_boundary: OptConfigType = None,
                 deep_sup_boundary_weight: float = 0.0,
                 deep_sup_fg_weight: float = 0.0,
                 instance_post_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            loss_align=loss_align,
            alignment_stage=alignment_stage,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.boundary_attention = None
        if boundary_attention is not None:
            self.boundary_attention = MODELS.build(boundary_attention)

        self.loss_boundary = None
        if loss_boundary is not None:
            self.loss_boundary = MODELS.build(loss_boundary)

        # Deep-supervision weights for the load-bearing module heads.
        # 0.0 keeps the legacy behaviour (no auxiliary loss even if a module
        # happens to emit deep-supervision predictions).
        self.deep_sup_boundary_weight = deep_sup_boundary_weight
        self.deep_sup_fg_weight = deep_sup_fg_weight

        self.instance_post_cfg = instance_post_cfg or {}

    @property
    def with_boundary_attention(self) -> bool:
        return self.boundary_attention is not None

    @property
    def with_boundary_loss(self) -> bool:
        return self.loss_boundary is not None

    # ------------------------------------------------------------------
    # Feature extraction with boundary modulation
    # ------------------------------------------------------------------

    def _extract_feat_with_boundary(
        self, inputs: Tensor
    ) -> Tuple[List[Tensor], Dict, Optional[Tensor]]:
        """Extract features, apply boundary attention, return everything."""
        x, aux_outputs = self._extract_feat_with_aux(inputs)

        boundary_pred = None
        if self.with_boundary_attention:
            rgb_feats = aux_outputs.get('rgb_feats')
            height_feats = aux_outputs.get('height_feats')
            bam_out = self.boundary_attention(
                tuple(x),
                rgb_feats=rgb_feats,
                height_feats=height_feats)
            if len(bam_out) == 3:
                x, boundary_pred, boundary_aux = bam_out
            else:  # backward compatibility with a 2-tuple BAM
                x, boundary_pred = bam_out
                boundary_aux = None
            x = list(x)
            aux_outputs['boundary_aux_preds'] = boundary_aux

        return x, aux_outputs, boundary_pred

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x, aux_outputs, boundary_pred = \
            self._extract_feat_with_boundary(inputs)

        losses = dict()

        # Decode head loss (Mask2Former cls + mask + dice via Hungarian)
        losses.update(self._decode_head_forward_train(x, data_samples))

        # Auxiliary head (if any)
        if self.with_auxiliary_head:
            losses.update(
                self._auxiliary_head_forward_train(x, data_samples))

        # Alignment loss (RGB-height feature alignment)
        align_feats = self._resolve_alignment_features(aux_outputs)
        if align_feats is not None:
            losses['loss_align'] = self.loss_align(*align_feats)

        # Boundary loss (main FPN-summed BAM head)
        boundary_gt = None
        if boundary_pred is not None and self.with_boundary_loss:
            boundary_gt = self._stack_boundary_gt(data_samples, inputs.device)
            losses['loss_boundary'] = self.loss_boundary(
                boundary_pred, boundary_gt)

        # ---- Deep-supervision auxiliary losses (load-bearing modules) ----
        # HBFM high-frequency branch and BAM per-scale evidence → boundary.
        hbfm_preds = aux_outputs.get('hbfm_boundary_preds')
        bam_aux = aux_outputs.get('boundary_aux_preds')
        if self.deep_sup_boundary_weight > 0 and (hbfm_preds or bam_aux):
            if boundary_gt is None:
                boundary_gt = self._stack_boundary_gt(
                    data_samples, inputs.device)
            if hbfm_preds:
                losses['loss_hbfm_boundary'] = self.deep_sup_boundary_weight \
                    * self._mean_aux_loss(hbfm_preds, boundary_gt)
            if bam_aux:
                losses['loss_bam_aux'] = self.deep_sup_boundary_weight \
                    * self._mean_aux_loss(bam_aux, boundary_gt)

        # MSCAM multi-scale branch → foreground (plot region).
        mscam_preds = aux_outputs.get('mscam_fg_preds')
        if self.deep_sup_fg_weight > 0 and mscam_preds:
            fg_gt = self._stack_fg_gt(data_samples, inputs.device)
            losses['loss_mscam_fg'] = self.deep_sup_fg_weight \
                * self._mean_aux_loss(mscam_preds, fg_gt)

        return losses

    # ------------------------------------------------------------------
    # Deep-supervision helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_boundary_gt(data_samples: SampleList,
                           device: torch.device) -> Tensor:
        """Stack per-sample binary boundary maps into ``(B, 1, H, W)``."""
        gts = []
        for ds in data_samples:
            if hasattr(ds, 'gt_boundary_map'):
                gts.append(ds.gt_boundary_map.data)
            else:
                h, w = ds.gt_sem_seg.data.shape[-2:]
                gts.append(torch.zeros(1, h, w, dtype=torch.long))
        return torch.stack(gts).to(device).float()

    @staticmethod
    def _stack_fg_gt(data_samples: SampleList,
                     device: torch.device) -> Tensor:
        """Stack per-sample foreground (plot) masks into ``(B, 1, H, W)``."""
        gts = []
        for ds in data_samples:
            seg = ds.gt_sem_seg.data  # (1, H, W), values in {0, 1, 255}
            gts.append((seg == 1).long())
        return torch.stack(gts).to(device).float()

    @staticmethod
    def _aux_bce(pred: Tensor, target: Tensor) -> Tensor:
        """Class-balanced BCE between a logit map and a dense target."""
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(
                target, size=pred.shape[-2:], mode='nearest')
        pos = target.sum().clamp(min=1.0)
        neg = (1.0 - target).sum().clamp(min=1.0)
        pos_weight = (neg / pos).clamp(max=100.0)
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight)

    def _mean_aux_loss(self, preds: Tuple[Tensor, ...],
                       target: Tensor) -> Tensor:
        """Mean of :meth:`_aux_bce` over a tuple of multi-scale logits."""
        return sum(self._aux_bce(p, target) for p in preds) / len(preds)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        x, _, _ = self._extract_feat_with_boundary(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        x, _, _ = self._extract_feat_with_boundary(inputs)
        seg_logits = self.decode_head.predict(
            x, batch_img_metas, self.test_cfg)
        return seg_logits

    def predict(self, inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Produce both semantic and instance predictions in a single pass.

        The semantic prediction is derived from instance masks (union of all
        instance masks → plot, elsewhere → background) rather than from the
        Mask2Former einsum aggregation, which treats background implicitly.
        """
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:])
            ] * inputs.shape[0]

        # Single backbone + BAM forward
        x, _, boundary_pred = self._extract_feat_with_boundary(inputs)

        # Instance prediction first — semantic is derived from it
        instance_results = None
        if hasattr(self.decode_head, 'predict_instances'):
            instance_results = self.decode_head.predict_instances(
                x, batch_img_metas, self.test_cfg)

        # Build semantic prediction from instance masks
        # Union of all predicted instance masks → class 1 (plot),
        # everything else → class 0 (background)
        if instance_results is not None:
            for i, (ds, inst) in enumerate(
                    zip(data_samples, instance_results)):
                ds.pred_instances = inst
                meta = ds.metainfo
                ori_h, ori_w = meta['ori_shape'][:2]
                if inst.masks.shape[0] > 0:
                    # Union of all instance binary masks
                    union = inst.masks.any(dim=0)  # (H, W) bool
                    sem_pred = union.long().unsqueeze(0)  # (1, H, W)
                else:
                    sem_pred = torch.zeros(
                        1, ori_h, ori_w,
                        device=inputs.device, dtype=torch.long)
                # Resize to ori_shape if needed
                cur_h, cur_w = sem_pred.shape[1:]
                if (cur_h, cur_w) != (ori_h, ori_w):
                    sem_pred = F.interpolate(
                        sem_pred.unsqueeze(0).float(),
                        size=(ori_h, ori_w),
                        mode='nearest').squeeze(0).long()
                ds.set_data({
                    'pred_sem_seg': PixelData(data=sem_pred),
                })
        else:
            # Fallback: use Mask2Former einsum aggregation
            seg_logits = self.decode_head.predict(
                x, batch_img_metas, self.test_cfg)
            data_samples = self.postprocess_result(seg_logits, data_samples)

        return data_samples

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x, _, _ = self._extract_feat_with_boundary(inputs)
        return self.decode_head.forward(x)
