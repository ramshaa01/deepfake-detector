# Day 37 Part A Notes: Seed=7 Training Run

## Purpose

The production model (`day32_finetuned_converged.pth`) was trained once from a single
initialization. Day 37 begins multi-seed variance reporting: training the identical
architecture and pipeline three times with different random seeds and reporting mean ± std
across seeds. This quantifies whether the reported 84.00% accuracy is a stable figure or
could be a lucky draw from a high-variance distribution.

This file documents the first additional seed: **seed=7**.

## What Is and Is Not Changed Between Seeds

| Component | Seed=7 run | Original Day 32 run |
|---|---|---|
| Architecture | EfficientNet-B0, pretrained ImageNet backbone | Identical |
| Training pipeline | Head-only then full fine-tune, differential LRs (1e-5 / 1e-4), ReduceLROnPlateau, early stopping patience=4 | Identical |
| Train/val/test splits | `data/splits/{train,val,test}.csv` — unchanged | Identical files |
| Augmentation | Identical (`get_train_transforms()`) | Identical |
| Random seed | **7** (model init, DataLoader shuffle) | Unrecorded (implicit default) |

## Training Curve Summary (Stage 2: Full Fine-Tune)

Early stopping fired at epoch 25 (patience=4, val_loss-based), with the best checkpoint
at **epoch 23** (val_acc=86.67%, val_loss=0.3268).

The training followed the same convergence pattern seen in Day 32:
- Early epochs: rapid val_acc improvement from head-only starting point
- Mid-run: backbone unlocking effect takes hold, val_acc climbs into 80s
- Best: epoch 23, val_acc 86.67% — slightly better than Day 32's training val_acc
- Early stopping: epoch 25 (2 epochs past best without val_loss improvement)

## Final Test-Set Results vs Day 32

| Metric | Seed=7 | Day 32 original | Delta |
|---|---|---|---|
| **Accuracy** | **84.33%** | **84.00%** | **+0.33 pp** |
| Precision | 0.8160 | 0.8088 | +0.0072 |
| Recall | 0.8867 | 0.7333 | +0.1534 |
| F1 Score | 0.8498 | 0.7692 | +0.0806 |
| **ROC-AUC** | **0.9324** | **0.9372** | **-0.0048** |

## Interpretation

Both seeds land at very similar accuracy (84.00% vs 84.33%, delta +0.33pp) and
ROC-AUC (0.9372 vs 0.9324, delta -0.0048). This is encouraging: the headline figures
are stable across seeds.

The larger apparent delta in Recall (0.7333 vs 0.8867, +0.1534) and F1 (0.7692 vs 0.8498)
at fixed threshold 0.5 reflects the expected sensitivity of threshold-dependent metrics
to small model differences. ROC-AUC is the appropriate threshold-independent comparison
and it is consistent across seeds (-0.0048, well within run-to-run noise for this dataset
size).

**Preliminary variance signal:** Both seeds produce ~84% accuracy and ~0.93 AUC.
Full mean±std will be computed after seed=123 (Part B) and seed=0 (Part C) complete.

## Checkpoints

| File | Contents |
|---|---|
| `model/checkpoints/day37_seed7_head_only_best.pth` | Best head-only Stage 1 checkpoint (epoch 10) |
| `model/checkpoints/day37_seed7_best.pth` | Best fine-tune checkpoint (epoch 23, val_acc=86.67%) |
| `model/checkpoints/day37_seed7_latest.pth` | Last epoch checkpoint (epoch 25) |
