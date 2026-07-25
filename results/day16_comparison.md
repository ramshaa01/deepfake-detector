# Day 16: Three-Way Model Comparison

This document compares the three primary model iterations evaluated on the identical 300-sample test set (150 real, 150 fake):

1. **Day 7**: Baseline Head-Only EfficientNet-B0
2. **Day 10**: Fine-Tuned EfficientNet-B0 (end-to-end unfreeze)
3. **Day 16**: CNN + FFT Frequency Feature Fusion

## Three-Way Comparison Table

| Metric | Day 7 (Head-Only) | Day 10 (Fine-Tuned CNN) | Day 16 (CNN + FFT Fusion) | Change (Day 16 vs Day 10) |
|---|---|---|---|---|
| **Accuracy** | 75.67% | 78.00% | **79.33%** | +1.33% |
| **ROC-AUC** | 0.8249 | 0.8492 | **0.8544** | +0.0052 |
| **Precision** | 75.16% | **78.00%** | 75.58% | -2.42% |
| **Recall** | 76.67% | 78.00% | **86.67%** | +8.67% |
| **F1 Score** | 75.91% | 78.00% | **0.8075** | +2.75% |
| **Avg Inference Time** | ~142 ms | ~58 ms | **~33 ms** | -25 ms |

*Note: Inference time variations reflect CPU execution environment variance across runs.*

## Analysis & Findings

- **Minor Overall Gain (+1.33% Accuracy, +0.0052 ROC-AUC)**: Fusing the 16-bin FFT radial energy feature with the frozen CNN backbone provided a marginal improvement in overall accuracy (79.33% vs 78.00%) and ROC-AUC (0.8544 vs 0.8492).
- **Shift Toward Higher Recall (+8.67%)**: The most noticeable shift introduced by the fusion head is a significantly higher Recall for the fake class (86.67% vs 78.00%), meaning the fusion model catches 130 out of 150 synthetic faces (only 20 false negatives vs 33 in Day 10). However, this came at a slight cost to Precision (75.58% vs 78.00%) due to an increase in false positives (42 vs 33).
- **Connection to Day 15 Findings**: On Day 15, we discovered that the standalone 1D radial FFT feature achieved only 54.33% accuracy (barely above random guessing), because StyleGAN2 has largely eliminated coarse global spectral artifacts and tight MTCNN face crops disrupt global grid frequency signatures. As hypothesized, combining a weak standalone signal with a strong deep feature extractor yields only minor incremental gains rather than a major breakthrough. 

## Conclusion
The fusion experiment validates our hypothesis: simple global frequency features provide only marginal complementary value when combined with a fine-tuned deep CNN. The fusion model pushes accuracy close to 80% and significantly boosts fake detection recall, providing a useful data point for multi-modal feature combination in synthetic face detection.
