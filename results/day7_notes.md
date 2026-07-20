# Day 7 Notes: Official Baseline Test-Set Evaluation

## Summary
The head-only EfficientNet-B0 model trained in Day 6 was evaluated on the isolated Test Set (300 images: 150 real, 150 fake). The metrics produced here represent the **Official Pre-Fine-Tuning Baseline**.

## Official Baseline Metrics
- **Test Accuracy**: 75.67%
- **Precision**: 75.16%
- **Recall**: 76.67%
- **F1 Score**: 75.91%
- **ROC-AUC**: 0.8249

*Note: The ROC-AUC of 0.825 indicates a strong underlying separation of the two classes even though hard-threshold accuracy is at ~76%. The model is balanced, favoring neither class excessively (Precision and Recall are nearly identical).*

## Performance
- **Average Inference Time**: 142.10 ms per image (on CPU). This translates to ~7 FPS on CPU, which is highly competitive for a real-time use case without hardware acceleration.

## Goals for Fine-Tuning (Day 8+)
To declare full fine-tuning a success, unfreezing the EfficientNet-B0 backbone must meaningfully surpass these baseline metrics. Specifically, we should look for:
- Accuracy pushing into the mid-80s or 90s.
- ROC-AUC climbing closer to 0.90+.

If the metrics plateau around 75% even after fine-tuning, it would suggest the model architecture or data pipeline is bottlenecking.

## Generated Artefacts
- `results/day7_baseline_metrics.md`: The raw metrics output.
- `results/day7_confusion_matrix.png`: Visual distribution of True/False Positives and Negatives.
- `results/day7_roc_curve.png`: The ROC curve showing the TPR vs FPR tradeoff.
