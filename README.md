# Flood Area Segmentation under Partial Supervision

## Overview
Segmentation experiment on satellite flood imagery (290 images) using U-Net 
with multiple encoders under sparse pixel supervision, implemented in PyTorch.

## Key Results
| Encoder | Drop Rate | IoU | Dice | F1 |
|---------|-----------|-----|------|----|
| ResNet18 | 0.1 | 0.612 | 0.734 | 0.798 |
| EfficientNet-B0 | 0.5 | 0.637 | 0.762 | 0.810 |
| Xception (Balanced) | 0.7 | **0.642** | **0.763** | **0.815** |

## Methodology
- **Model:** U-Net with 3 encoders (ResNet18, EfficientNet-B0, Xception) — ImageNet pretrained
- **Loss:** Custom Spatial Focal Loss with pixel-level supervision mask
- **Training:** 3-Fold Cross Validation, Early Stopping
- **Supervision:** Random and Class-Balanced pixel sampling strategies (10%--70% drop)
- **Evaluation:** Global pixel-wise (Precision, Recall, F1) + Per-image spatial (IoU, Dice)

## Key Findings
- EfficientNet-B0 generalizes better than larger encoders on small datasets
- Gamma=5 produces lower focal loss values but worse segmentation metrics
- Class-balanced sampling improves performance at low drop rates
- High drop rates act as regularization without significant performance degradation

## Files
- `notebook.ipynb` — Full training pipeline
- `flood_segmentation_report.pdf` — Detailed experiment report
