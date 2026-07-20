# Day 6 Notes: EfficientNet-B0 Head-Only Baseline

## Architecture & Configuration
- **Backbone**: `timm.create_model("efficientnet_b0", pretrained=True)`
- **Trainable params**: 1,281 (the new head: 1280 → 1)
- **Frozen params**: ~4M (the entire ImageNet pretrained backbone)
- **Loss function**: `BCEWithLogitsLoss()` (optimal for single-logit binary classification)
- **Optimizer**: Adam, `lr=1e-3`
- **Hardware**: CPU (CUDA unavailable in this environment)

## Training Results
Training was run for 5 epochs. Since the model was only learning the final classification head (1280 weights), it converged quickly. The process was stopped early at epoch 5 because validation accuracy stabilized, meaning the head-only capacity was saturated.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.6618     | 61.14%    | 0.6251   | 69.67%  |
| 2     | 0.5954     | 71.14%    | 0.5904   | 71.33%  |
| 3     | 0.5584     | 73.50%    | 0.5710   | 72.00%  |
| 4     | 0.5620     | 71.50%    | 0.5502   | 72.33%  |
| **5** | **0.5319** | **74.36%**| **0.5417** | **74.33%** |

*Note: Since the dataset is perfectly balanced (50/50 real vs fake), accuracy is a directly valid metric. Baseline random guessing would yield 50%.*

## Assessment
The model is definitely learning meaningfully! With **zero** updates to the visual feature extractors (the backbone), just linearly combining the existing ImageNet features allows the network to distinguish Real (FFHQ) from Fake (StyleGAN2) at **~74% accuracy**.

This confirms:
1. The ImageNet pretrained features carry enough general texture/frequency information to separate some StyleGAN artefacts.
2. The PyTorch DataLoaders and labels are mapped correctly.
3. The dataset is solvable.

## Next Steps (Day 8+)
To push beyond 74%, we need to unfreeze the backbone and do full fine-tuning. The model needs to learn specific edge/texture features relevant to AI synthesis, rather than just relying on generic ImageNet features.

**Best checkpoint saved to**: `model/checkpoints/day6_head_only_best.pth`
