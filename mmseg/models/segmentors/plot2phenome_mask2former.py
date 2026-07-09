# Copyright (c) OpenMMLab. All rights reserved.
"""Plot2Phenome segmentor built on top of MMSeg Mask2Former."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class Plot2PhenomeMask2Former(EncoderDecoder):
    """Encoder-decoder segmentor with an additional RGB-height alignment loss."""

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_align: OptConfigType = None,
                 alignment_stage: int = 2,
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
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.alignment_stage = alignment_stage
        self.loss_align = MODELS.build(loss_align) if loss_align else None

    @property
    def with_alignment_loss(self) -> bool:
        return self.loss_align is not None

    def _extract_feat_with_aux(self, inputs: Tensor) -> Tuple[List[Tensor],
                                                              Dict]:
        backbone_outputs = self.backbone(inputs)
        aux_outputs: Dict = {}
        if isinstance(backbone_outputs, dict):
            aux_outputs = {
                key: value
                for key, value in backbone_outputs.items()
                if key != 'features'
            }
            x = list(backbone_outputs['features'])
        else:
            x = list(backbone_outputs)
        if self.with_neck:
            x = self.neck(x)
        return x, aux_outputs

    def _resolve_alignment_features(self, aux_outputs: Dict) -> Optional[Tuple]:
        if not self.with_alignment_loss:
            return None
        rgb_feats = aux_outputs.get('rgb_feats')
        height_feats = aux_outputs.get('height_feats')
        if rgb_feats is None or height_feats is None:
            return None
        stage_idx = self.alignment_stage
        if stage_idx < 0:
            stage_idx = len(rgb_feats) + stage_idx
        return rgb_feats[stage_idx], height_feats[stage_idx]

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        x, _ = self._extract_feat_with_aux(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        x, _ = self._extract_feat_with_aux(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x, aux_outputs = self._extract_feat_with_aux(inputs)
        losses = dict()
        losses.update(self._decode_head_forward_train(x, data_samples))

        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(x, data_samples))

        align_feats = self._resolve_alignment_features(aux_outputs)
        if align_feats is not None:
            losses['loss_align'] = self.loss_align(*align_feats)
        return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x, _ = self._extract_feat_with_aux(inputs)
        return self.decode_head.forward(x)
