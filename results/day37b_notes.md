# Day 37 Part B Notes: Seed=123 Training Run

## Purpose

To complete the 3-run variance report, this file documents the third and final seed (**seed=123**) training run of the production EfficientNet-B0 architecture. We compare its metrics against both the original Day 32 run and the Seed 7 run from Day 37 Part A.

## Training Improvements (Live Logging)

During Part A, we discovered that `print` statements in the training script were not being flushed to standard output in real time due to Python's default buffering, rendering the background task logs empty until the end of the run.

In Part B, we implemented real-time monitoring improvements in `train_day37_seed123.py`:
- Flushed standard output automatically using the `-u` flag.
- Wrote incremental CSV logs directly to disk after every epoch.
- Logged the Early Stopping (ES) counter to easily track convergence progress in the logs.

## Training Curve Summary (Stage 2: Full Fine-Tune)

The training for seed=123 followed the established convergence pattern:
- **Best Checkpoint:** Epoch 24 (val_acc=86.33%, val_loss=0.3309).
- **Early Stopping:** Triggered at epoch 29 after 4 epochs of no val_loss improvement.

## Final Test-Set Results Comparison

| Metric | Seed=123 | Seed=7 | Day 32 original | Delta (Seed 123 vs Day 32) |
|---|---|---|---|---|
| **Accuracy** | **84.67%** | 84.33% | **84.00%** | **+0.67 pp** |
| Precision | 0.8171 | 0.8160 | 0.8088 | +0.0083 |
| Recall | 0.8933 | 0.8867 | 0.7333 | +0.1600 |
| F1 Score | 0.8535 | 0.8498 | 0.7692 | +0.0843 |
| **ROC-AUC** | **0.9267** | 0.9324 | **0.9372** | **-0.0105** |

## Interpretation

The test accuracy remains remarkably stable across all three seeds, hovering around 84% (84.00%, 84.33%, and 84.67%). This confirms the validity of our original ~84% claim. The ROC-AUC score is also extremely consistent, remaining tightly clustered in the 0.926 - 0.937 range.

The significant fluctuations in Recall and F1 Score at the default decision threshold (0.5) across seeds highlight why ROC-AUC (which evaluates all possible thresholds) is a far more robust metric for tracking genuine model quality.

With all three seeds trained and evaluated, we are now ready to compute the final mean ± standard deviation for the production model in Part C.

## Checkpoints

| File | Contents |
|---|---|
| `model/checkpoints/day37_seed123_head_only_best.pth` | Best head-only Stage 1 checkpoint |
| `model/checkpoints/day37_seed123_best.pth` | Best fine-tune checkpoint (epoch 24, val_acc=86.33%) |
| `model/checkpoints/day37_seed123_latest.pth` | Last epoch checkpoint (epoch 29) |
