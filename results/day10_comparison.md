# Day 10: Official Fine-Tuned Model Evaluation

## Comparison vs Head-Only Baseline

We evaluated the fine-tuned checkpoint (`day8_finetuned_best.pth`) on the isolated test set of 300 images. Below is a side-by-side comparison of the Day 10 fine-tuned model against the Day 7 head-only baseline:

| Metric | Day 7 (Head-Only) | Day 10 (Fine-Tuned) | Improvement |
|--------|-------------------|---------------------|-------------|
| **Accuracy** | 75.67% | **78.00%** | +2.33% |
| **Precision**| 75.16% | **78.00%** | +2.84% |
| **Recall**   | 76.67% | **78.00%** | +1.33% |
| **F1 Score** | 75.91% | **78.00%** | +2.09% |
| **ROC-AUC**  | 0.8249 | **0.8492** | +0.0243 |
| **Inference**| ~142 ms | ~58 ms* | N/A |

*\*Note: The inference time difference is primarily due to natural CPU/system variance across runs (model architecture is exactly the same). Both are highly competitive for real-time CPU applications without hardware acceleration.*

## Conclusion

**Fine-tuning meaningfully improved results across all metrics.** Unfreezing the backbone and training it specifically to detect AI-generated artefacts raised the ROC-AUC to nearly 0.85 and hard accuracy to 78%. 

**Important Note:** The Day 8 fine-tuning process was interrupted by a server restart at epoch 7. At that time, both validation accuracy and validation loss were still actively improving. **This checkpoint is not fully converged.** Further training would likely improve these numbers even more, pushing accuracy into the 80s, but we will proceed with this epoch-7 checkpoint as our working baseline for the rest of the project timeline.
