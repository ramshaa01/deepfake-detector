# Model Checkpoints Registry

This directory stores trained weights for the AI-Generated Face Detector project. 

> **Note:** `.pth` weight files are excluded from Git via `.gitignore` to keep the repository lightweight. Model state definitions and evaluation scripts remain fully reproducible.

---

## Checkpoints Log

| Checkpoint File | Stage / Description | Test Accuracy | ROC-AUC | Status |
|---|---|---|---|---|
| `day6_head_only_best.pth` | Day 6: EfficientNet-B0 linear head training (frozen backbone) | 75.67% | 0.8249 | Legacy Baseline |
| `day8_finetuned_best.pth` | Day 8–10: End-to-end fine-tuned EfficientNet-B0 (interrupted) | 78.00% | 0.8492 | Legacy Baseline |
| `day16_fusion_best.pth` | Day 16: EfficientNet-B0 + 16-bin FFT Radial Energy Fusion MLP | 79.33% | 0.8544 | Evaluated, rejected |
| `day34_xception_best.pth`| Day 34: XceptionNet field-standard baseline | 81.60% | 0.9120 | Baseline, underperforms |
| `day37_seed7_best.pth` | Day 37a: EfficientNet-B0 (converged, seed 7) | 84.33% | 0.9324 | Evaluated for variance |
| `day37_seed123_best.pth` | Day 37b: EfficientNet-B0 (converged, seed 123) | 84.67% | 0.9267 | Evaluated for variance |
| **`day32_finetuned_converged.pth`** | **Day 32: EfficientNet-B0 (converged, original seed)** | **84.00%** | **0.9372** | **FINAL / PRODUCTION** |

---

## Final Model Designation

**Selected Production Checkpoint:** `model/checkpoints/day32_finetuned_converged.pth`

### Selection Rationale
Following the Day 32 matched-operating-point analysis and Day 37 multi-seed variance testing (`results/day37_multiseed_variance.md`), the **CNN-only model** was confirmed as the final production architecture.

- **High Stability**: 3 independent training runs confirmed an average accuracy of **84.33% ± 0.34%** and ROC-AUC of **0.9321 ± 0.0053**.
- **Performance Trade-Off (vs Fusion)**: The CNN-only model strictly dominates the fusion model, achieving higher ROC-AUC (0.9372 vs 0.9328) and faster inference (73.5 ms vs 88.05 ms).
- **Decision Rule**: The original Day 32 checkpoint (`day32_finetuned_converged.pth`) remains the active deployed model on Render, as it represents a fully valid draw from the stable distribution with the highest ROC-AUC.
