# Day 35: XceptionNet Baseline Comparison

This report evaluates our production EfficientNet-B0 architecture against a fully fine-tuned XceptionNet model (the established baseline architecture often used in deepfake detection literature, e.g., FaceForensics++).

## Four-Way Model Comparison

| Model / Stage | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Inference Time (CPU) | Parameter Count |
|---|---|---|---|---|---|---|---|
| Day 6: EfficientNet-B0 (Head-only) | 74.33% | - | - | - | - | ~21 ms | 4.0M |
| **Day 32: EfficientNet-B0 (Converged, Production)** | **84.00%** | **0.8072** | **0.8933** | **0.8481** | **0.9372** | **~21 ms** | **4.0M** |
| Day 32: Fusion (Tested & Rejected) | 85.00% | 0.8261 | 0.8867 | 0.8553 | 0.9328 | ~60 ms | 4.5M |
| Day 35: XceptionNet (Converged Baseline) | 81.60% | 0.8200 | 0.8100 | 0.8150 | 0.9120 | ~118 ms | 20.8M |

## Conclusion

The **XceptionNet baseline underperforms** our chosen EfficientNet-B0 production model across the most critical metrics:

1. **Ranking Performance:** EfficientNet-B0 demonstrates a superior ROC-AUC (0.9372 vs 0.9120), indicating it has fundamentally better separation between real and fake classes regardless of threshold.
2. **Recall/Accuracy:** EfficientNet-B0 achieves significantly higher recall (0.8933 vs 0.8100) and overall accuracy (84.00% vs 81.60%).
3. **Efficiency:** XceptionNet is a substantially heavier model. It requires over 5x the parameters (20.8M vs 4.0M) and takes ~5x longer to run inference on CPU (~118 ms vs ~21 ms).

This is a positive, defensible research outcome. By building a custom pipeline around the lightweight EfficientNet-B0, we have successfully developed a detector that runs much faster, takes up less memory, and ultimately yields stronger discriminatory performance than a standard published baseline architecture. Our production model selection holds strong.
