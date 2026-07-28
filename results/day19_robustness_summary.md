# Day 19: Robustness Evaluation

Evaluation of the final fusion model (`day16_fusion_best.pth`) across various image perturbations.

| Condition | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **clean (baseline)** | 79.33% | 0.7558 | 0.8667 | 0.8075 | 0.8544 |
| jpeg_q90 | 72.67% | 0.6771 | 0.8667 | 0.7602 | 0.8268 |
| jpeg_q70 | 68.67% | 0.6359 | 0.8733 | 0.7360 | 0.7727 |
| jpeg_q50 | 65.33% | 0.6150 | 0.8200 | 0.7029 | 0.7358 |
| jpeg_q30 | 65.67% | 0.6313 | 0.7533 | 0.6869 | 0.7112 |
| blur_s1 | 62.67% | 0.6131 | 0.6867 | 0.6478 | 0.6466 |
| blur_s2 | 52.33% | 0.5146 | 0.8200 | 0.6324 | 0.5363 |
| blur_s4 | 54.00% | 0.5213 | 0.9800 | 0.6806 | 0.5192 |
| resize_0.5x | 65.33% | 0.6106 | 0.8467 | 0.7095 | 0.7128 |
| resize_0.25x | 52.33% | 0.5131 | 0.9133 | 0.6571 | 0.5244 |
