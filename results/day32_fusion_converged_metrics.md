# Day 32: Converged Fusion Model Test Set Metrics

**Checkpoint:** `model/checkpoints/day32_fusion_converged.pth`
**CNN Backbone:** `day32_finetuned_converged.pth` (trained to full convergence)
**Comparison:** Day 16 (`day16_fusion_best.pth`, based on interrupted day8 backbone)

## Direct Comparison: Day 32 (Converged) vs Day 16 (Interrupted backbone)

| Metric | Day 16 (Official) | Day 32 (Converged) | Delta |
|---|---|---|---|
| Accuracy | 79.33% | 85.00% | +5.67% |
| Precision | 0.7558 | 0.8261 | +0.0703 |
| Recall | 0.8667 | 0.8867 | +0.0200 |
| F1 Score | 0.8075 | 0.8553 | +0.0478 |
| ROC-AUC | 0.8544 | 0.9328 | +0.0784 |
| PR-AUC | 0.8222 | 0.9273 | +0.1051 |
| Avg Infer. Time | ~241 ms | 60.22 ms | — |

## Operating Point Analysis (Fusion vs CNN-only, matched recall)

| Target Recall | Model | Actual Recall | Precision | Threshold |
|---|---|---|---|---|
| 0.800 | Fusion | 0.8000 | 0.8889 | 0.6424 |
| 0.800 | CNN-only | 0.8000 | 0.9231 | 0.8077 |
| 0.850 | Fusion | 0.8467 | 0.8581 | 0.5831 |
| 0.850 | CNN-only | 0.8467 | 0.8699 | 0.6866 |
| 0.867 | Fusion | 0.8667 | 0.8387 | 0.5555 |
| 0.867 | CNN-only | 0.8667 | 0.8609 | 0.6553 |
| 0.900 | Fusion | 0.9000 | 0.8036 | 0.4784 |
| 0.900 | CNN-only | 0.9000 | 0.7988 | 0.4858 |

## Confusion Matrix

```
[[122  28]
 [ 17 133]]
```

*(Rows: True. Cols: Predicted. 0=Real, 1=Fake)*
