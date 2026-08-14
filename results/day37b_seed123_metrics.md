# Day 37 Part B: Seed=123 Test Metrics

**Checkpoint:** `model/checkpoints/day37_seed123_best.pth`
**Architecture:** EfficientNet-B0 (CNN-only, identical to production Day 32 architecture)
**Evaluation set:** `data/splits/test.csv` (300 images, balanced 150 real / 150 fake)
**Face detector:** MTCNN (matching official Day 32 methodology — not Haar Cascade)
**Seed:** 123 (initialization + DataLoader shuffle; train/val/test splits unchanged)

## Test-Set Metrics

| Metric | Seed=123 (Day 37b) | Seed=7 (Day 37a) | Original Day 32 |
|---|---|---|---|
| **Accuracy** | **84.67%** | 84.33% | 84.00% |
| Precision | 0.8171 | 0.8160 | 0.8088 |
| Recall | 0.8933 | 0.8867 | 0.7333 |
| F1 Score | 0.8535 | 0.8498 | 0.7692 |
| **ROC-AUC** | **0.9267** | 0.9324 | 0.9372 |

## Training Summary

| Stage | Detail |
|---|---|
| Stage 1 (head-only) | 10 epochs, lr=1e-3, pretrained backbone frozen |
| Stage 2 (full fine-tune) | Max 50 epochs, early stopping patience=4 on val_loss |
| Best checkpoint epoch | 24 |
| Best val accuracy | 86.33% |
| Best val loss | 0.3309 |
| Stopped at epoch | 29 (early stopping fired 4 epochs after best val loss) |
| Avg inference time | ~26 ms / image (batch=1, CPU) |

## Plots

- `results/day37b_seed123_confusion_matrix.png`
- `results/day37b_seed123_roc_curve.png`
