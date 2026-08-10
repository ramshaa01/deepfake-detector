# Day 34a Notes: XceptionNet Head-Only Training

## Training Process
- **Architecture:** XceptionNet (`legacy_xception` from `timm`) pretrained on ImageNet.
- **Modifications:** Replaced the final 1000-class `fc` layer with a binary classification layer (`Linear(2048 -> 1)`).
- **Training Strategy:** Backbone frozen. Head-only training using Adam optimizer (lr=1e-3).
- **Data Pipeline:** Identical to the EfficientNet-B0 production model pipeline, utilizing the same `DeepfakeFaceDataset`, exactly the same `data/splits/*.csv` files, and identical augmentation/normalization (ImageNet mean/std). The only difference is the input size (resized to 299x299 as expected by Xception).

## Results
The head-only training ran for 7 epochs before early stopping criteria were met on `val_loss`.

- **Final Train Accuracy:** 75.12%
- **Final Train Loss:** 0.5211
- **Final Val Accuracy:** 74.67% (Peak at Epoch 6)
- **Final Val Loss:** 0.5482 (Best at Epoch 6)

## Observations
The model is learning meaningfully. The validation accuracy of ~74.67% is well above the random 50% baseline for this perfectly balanced dataset. This mirrors the behavior we saw during Day 6 when training the head of the EfficientNet-B0 model, confirming that the new classification head is properly warming up and capturing signals from the frozen Xception features.

The checkpoint is saved at `model/checkpoints/day34a_xception_headonly_best.pth`. Next step (Day 34b) is full end-to-end fine-tuning.
