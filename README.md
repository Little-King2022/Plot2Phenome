# Plot2Phenome

**Plot2Phenome: A UAV-Based Deep Learning Framework for Automated Micro-Plot Segmentation and Precision Phenotyping**

---

## Pre-Release: Core Model Architecture

This repository contains the **core model architecture** of RGBD-Plotformer — a dual-modal (RGB + DSM/Height) instance segmentation model for breeding plot segmentation from UAV imagery. The full runnable training/evaluation pipeline will be released upon paper acceptance.

---

## Core Modules

```
mmseg/models/
├── backbones/
│   └── dual_stream_resnet.py          # Dual-stream ResNet-50 backbone with stage-wise fusion
├── decode_heads/
│   └── plot2phenome_instance_head.py  # Mask2Former head with instance-level Hungarian matching
├── losses/
│   ├── alignment_loss.py              # Cross-modal cosine alignment loss
│   └── instance_boundary_loss.py      # BCE + Dice boundary supervision
├── segmentors/
│   ├── plot2phenome_mask2former.py    # Semantic segmentation segmentor
│   └── plot2phenome_instance_segmentor.py  # Instance segmentation segmentor with BAM
└── utils/
    ├── hbfm.py                        # Hierarchical Bi-modal Fusion (wavelet-like decomposition)
    ├── mscam.py                       # Multi-Scale Channel Attention
    ├── cmgfa.py                       # Cross-Modal Gated Fusion Attention
    └── boundary_attention.py          # Boundary Attention Module (BAM v1/v2)
```

### Architecture Overview

- **DualStreamResNet**: Dual-branch backbone with independent RGB and Height encoders, progressive stage-wise fusion, and cross-modal feature re-injection
- **HBFM**: Fuses RGB and wavelet-decomposed height features via `concat → 1×1 → 3×3DW → 1×1 + residual`
- **MSCAM**: Per-scale multi-scale channel-spatial attention for fused feature refinement
- **CMGFA**: Alternative fusion with bidirectional cross-modal gating and difference-aware refinement
- **BAM**: Dual-modal gradient attention for boundary-aware feature enhancement
- **AlignmentLoss**: Cosine similarity loss between GAP-pooled RGB and height features (λ=0.02)
- **InstanceBoundaryLoss**: Weighted BCE + Dice loss on predicted vs. ground-truth boundaries (λ=0.5)

### Key Results (40k iterations)

| Configuration | mIoU | AP@50 | PQ | Boundary F1 |
|---|---:|---:|---:|---:|
| Full Model (RGB+DSM, HBFM+MSCAM+BAM+Align) | **91.08** | 94.60 | **89.70** | **83.22** |
| RGB-only | 88.31 | 92.58 | 87.08 | 73.44 |

---

## Dependencies

- PyTorch ≥ 1.13
- `mmcv` ≥ 2.0.0
- `mmengine` ≥ 0.5.0
- `mmdet` (for Mask2Former components)
- `mmsegmentation` ≥ 1.0.0

---

## Citation

```bibtex
@article{plot2phenome,
  title={Plot2Phenome: A UAV-Based Deep Learning Framework for Automated Micro-Plot Segmentation and Precision Phenotyping},
  author={},
  journal={},
  year={}
}
```

---

## License

This project is licensed under the terms provided in the [LICENSE](LICENSE) file.

---

> **Note**: The full runnable code including training pipeline, dataset processing, evaluation scripts, and pre-trained weights will be released upon paper acceptance.
