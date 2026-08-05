# Day 31: Production Detector Delta (MTCNN vs Haar Cascade)

This evaluates the real-world deployment accuracy drop caused by using the lightweight Haar Cascade detector instead of MTCNN (due to 512MB RAM constraints on the Render free tier).

**Test Set:** 300 images (balanced).
**Haar Cascade Detection Failures (400 Bad Request):** 14 (4.7%)
*Note: Detection failures are treated as incorrect predictions for the Haar metrics to reflect true system accuracy.* 

| Metric | Day 16 (MTCNN - Official) | Day 31 (Haar Cascade - Production) | Delta |
|---|---|---|---|
| Accuracy | 79.33% | 73.00% | -6.33% |
| Precision | 0.7558 | 0.7018 | -0.0540 |
| Recall | 0.8667 | 0.8000 | -0.0667 |
| F1 Score | 0.8075 | 0.7477 | -0.0598 |
| ROC-AUC | 0.8544 | 0.7527 | -0.1017 |
