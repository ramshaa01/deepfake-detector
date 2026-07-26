# Model Checkpoints Registry

This directory stores trained weights for the AI-Generated Face Detector project. 

> **Note:** `.pth` weight files are excluded from Git via `.gitignore` to keep the repository lightweight. Model state definitions and evaluation scripts remain fully reproducible.

---

## Checkpoints Log

| Checkpoint File | Stage / Description | Test Accuracy | ROC-AUC | PR-AUC | Single-Image Latency | Status |
|---|---|---|---|---|---|---|
| `day6_head_only_best.pth` | Day 6: EfficientNet-B0 linear head training (frozen backbone) | 75.67% | 0.8249 | 0.7912 | ~195 ms | Legacy Baseline |
| `day8_finetuned_best.pth` | Day 8–10: End-to-end fine-tuned EfficientNet-B0 | 78.00% | 0.8492 | 0.8196 | ~191 ms | CNN-Only Baseline |
| `day16_fusion_best.pth` | Day 16: EfficientNet-B0 + 16-bin FFT Radial Energy Fusion MLP | **79.00%** | **0.8535** | **0.8222** | ~241 ms | **FINAL / PRODUCTION** |

---

## Final Model Designation

**Selected Production Checkpoint:** `model/checkpoints/day16_fusion_best.pth`

### Selection Rationale
Following the Day 16–17 benchmark investigation (`results/day16-17_timing_investigation.md`), the **CNN + FFT Fusion model** was selected as the final production architecture for all downstream deliverables (FastAPI serving, React frontend integration, and robustness testing).

- **Performance Trade-Off**: Fusing the frequency feature branch provides a small but genuine ranking and precision advantage at matched operating points (**+2.11% precision** at 86.67% recall, **0.8222 vs 0.8196 PR-AUC**, **79.00% vs 78.00% accuracy**).
- **Latency Overhead**: Adds **+50 ms latency** per single-image CPU inference (~241 ms vs ~191 ms).
- **Decision Rule**: For synthetic face screening in content moderation and identity verification pipelines, **classification accuracy and precision are prioritized over a 50 ms latency difference**.
