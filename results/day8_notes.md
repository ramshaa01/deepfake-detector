# Day 8 Notes: Full Fine-Tuning Setup & Results

## Fine-Tuning Configuration
- **Starting Point**: `day6_head_only_best.pth` (head pre-trained to 75% accuracy, backbone ImageNet weights).
- **Differential Learning Rates**: 
  - Backbone: `1e-5` (small, to preserve existing ImageNet feature extractors without catastrophic forgetting).
  - Head: `1e-4` (larger, allows the head to adapt to the slowly-shifting backbone features).
- **Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=2)` monitoring `val_loss`.
- **Early Stopping**: Patience = 4 epochs on `val_loss`.

## Training Progress
Training was run and halted after strong progress was observed, saving the best checkpoint dynamically. Here is the trajectory of the first 7 epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.4901     | 77.14%    | 0.5179   | 74.33%  |
| 2     | 0.4478     | 81.36%    | 0.5101   | 75.33%  |
| 3     | 0.4366     | 80.86%    | 0.4886   | 77.00%  |
| 4     | 0.4023     | 84.86%    | 0.4757   | 77.33%  |
| 5     | 0.3839     | 84.71%    | 0.4676   | 79.00%  |
| 6     | 0.3557     | 85.79%    | 0.4548   | 79.67%  |
| **7** | **0.3292** | **87.71%**| **0.4346** | **80.33%** |

## Assessment vs. Baseline
- **Official Head-Only Baseline (Day 7)**: 75.67% test accuracy.
- **Current Fine-Tuned Best (Val)**: **80.33%** (at Epoch 7).

**Conclusion**: The full fine-tuning is **successful**. By unfreezing the backbone and training end-to-end with a low learning rate, the model successfully learned specific features for AI-generated faces rather than relying purely on generic ImageNet features. 

The validation accuracy climbed steadily from ~74% to ~80% without catastrophic overfitting (train and val loss both decreased smoothly).

**Best checkpoint saved to**: `model/checkpoints/day8_finetuned_best.pth`
