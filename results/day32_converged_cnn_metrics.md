# Day 32: Converged CNN Test Set Metrics

**Checkpoint:** `model/checkpoints/day32_finetuned_converged.pth`
**Comparison baseline:** Day 10 (day8_finetuned_best.pth, interrupted at epoch 2)

## Direct Comparison: Day 32 (Converged) vs Day 10 (Interrupted)

| Metric | Day 10 (Interrupted) | Day 32 (Converged) | Delta |
|---|---|---|---|
| Accuracy | 78.00% | 84.00% | +6.00% |
| Precision | 0.7800 | 0.8072 | +0.0272 |
| Recall | 0.7800 | 0.8933 | +0.1133 |
| F1 Score | 0.7800 | 0.8481 | +0.0681 |
| ROC-AUC | 0.8492 | 0.9372 | +0.0880 |
| Avg Infer. Time | — | 21.36 ms | — |

## Interpretation

Full convergence **improved** on CNN-only accuracy relative to the interrupted Day 8 checkpoint.
Accuracy delta: +6.00%. ROC-AUC delta: +0.0880. F1 delta: +0.0681.

Note: Both numbers are CNN-only baselines. The fusion model evaluation follows separately.

## Confusion Matrix

```
[[118  32]
 [ 16 134]]
```

*(Rows: True. Cols: Predicted. 0=Real, 1=Fake)*
