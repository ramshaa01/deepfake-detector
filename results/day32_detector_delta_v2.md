# Day 32: Production Detector Delta (CNN‑only vs Haar Cascade)

This evaluates the real‑world deployment accuracy drop caused by using the lightweight Haar Cascade detector with the converged CNN‑only model.

**Test Set:** 300 images (balanced).
**Haar Cascade Detection Failures (400 Bad Request):** 14 (4.7%)
*Note: Detection failures are treated as incorrect predictions for the Haar metrics to reflect true system accuracy.*

| Metric | Day 16 (MTCNN – Official) | Day 32 (Haar Cascade – CNN‑only) | Delta |
|---|---|---|---|
| Accuracy | 79.33% | 78.00% | -1.33% |
| Precision | 0.7558 | 0.8088 | +0.0530 |
| Recall | 0.8667 | 0.7333 | -0.1334 |
| F1 Score | 0.8075 | 0.7692 | -0.0383 |
| ROC-AUC | 0.8544 | 0.8387 | -0.0157 |
