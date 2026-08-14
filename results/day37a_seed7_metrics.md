# Day 37 Part A: Seed=7 Test Metrics

**Checkpoint:** `model/checkpoints/day37_seed7_best.pth`
**Architecture:** EfficientNet-B0 (CNN-only, identical to production Day 32 architecture)
**Evaluation set:** `data/splits/test.csv` (300 images, balanced 150 real / 150 fake)
**Face detector:** MTCNN (matching official Day 32 methodology — not Haar Cascade)
**Seed:** 7 (initialization + DataLoader shuffle; train/val/test splits unchanged)

## Test-Set Metrics

| Metric | Seed=7 (Day 37a) | Original Day 32 (seed implicit) | Delta |
|---|---|---|---|
| **Accuracy** | **84.33%** | **84.00%** | **+0.33 pp** |
| Precision | 0.8160 | 0.8088 | +0.0072 |
| Recall | 0.8867 | 0.7333 | +0.1534 |
| F1 Score | 0.8498 | 0.7692 | +0.0806 |
| **ROC-AUC** | **0.9324** | **0.9372** | **-0.0048** |

## Training Summary

| Stage | Detail |
|---|---|
| Stage 1 (head-only) | 10 epochs, lr=1e-3, pretrained backbone frozen |
| Stage 2 (full fine-tune) | Max 50 epochs, early stopping patience=4 on val_loss |
| Best checkpoint epoch | 23 |
| Best val accuracy | 86.67% |
| Best val loss | 0.3268 |
| Stopped at epoch | 25 (early stopping fired 2 epochs after best) |
| Device | CPU |
| Avg inference time | 113.02 ms / image (batch=1, CPU) |

## Plots

- `results/day37a_seed7_confusion_matrix.png`
- `results/day37a_seed7_roc_curve.png`
