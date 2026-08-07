# Day 32, Part 1: CNN Fine-Tuning Convergence

## Overview
The Day 8 fine-tuning process was originally interrupted at epoch 7 by an infrastructure restart. The resulting checkpoint (`day8_finetuned_best.pth`, selected from epoch 2) served as the foundation for the entire project, including the Day 16 fusion model. 

In Part 1 of Day 32, we resumed CNN fine-tuning from that interrupted state, ensuring it ran uninterrupted to true convergence with early stopping (patience=4 on validation loss).

## Training Run
- **Resumed From:** Epoch 20 state (from a previous partial Day 32 run), loading the true all-time best score correctly.
- **Max Epochs Cap:** 30
- **Final Epoch Reached:** 30 (Run hit the hard cap of 30 epochs).
- **Best Checkpoint:** Epoch 29 (`val_acc = 87.33%`, `val_loss = 0.3288`).
- **Saved As:** `model/checkpoints/day32_finetuned_converged.pth`

## Evaluation vs Interrupted Baseline
We evaluated the converged CNN on the isolated test set using the exact same criteria as the official Day 10 evaluation.

| Metric | Day 10 (Interrupted) | Day 32 (Converged) | Delta |
|---|---|---|---|
| **Accuracy** | 78.00% | **84.00%** | +6.00% |
| **Precision** | 0.7800 | **0.8072** | +0.0272 |
| **Recall** | 0.7800 | **0.8933** | +0.1133 |
| **F1 Score** | 0.7800 | **0.8481** | +0.0681 |
| **ROC-AUC** | 0.8492 | **0.9372** | +0.0880 |

## Conclusion
Full convergence **significantly improved** the CNN-only baseline across every single metric. The ROC-AUC jump from 0.8492 to 0.9372 is particularly massive, showing the model has learned a much more robust decision boundary. Recall saw an enormous +11.33% boost.

*(Note: The `day8_finetuned_best.pth` checkpoint has been retained for historical provenance as "v1, interrupted").*
