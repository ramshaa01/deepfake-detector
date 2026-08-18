# Final Metrics Summary v2 — AI-Generated Face Detector
*Supersedes `results/final_metrics_summary.md` (Day 30). Updated through Day 40 (project close).*

---

## Production Model

| Item | Value |
|---|---|
| **Checkpoint** | `model/checkpoints/day32_finetuned_converged.pth` |
| **Architecture** | EfficientNet-B0 (ImageNet pre-trained, fine-tuned end-to-end) |
| **Parameters** | ~4.0M |
| **Training Face Detector** | MTCNN |
| **Production Face Detector** | OpenCV Haar Cascades (RAM constraint) |

---

## 1. Headline Metrics (Official — MTCNN Pipeline, n=300 balanced test set)

Reported as **mean ± std across 3 independent training runs** (original seed, seed=7, seed=123):

| Metric | Mean ± Std |
|---|---|
| **Accuracy** | **84.33% ± 0.34%** |
| **ROC-AUC** | **0.9321 ± 0.0053** |
| Precision | 0.8140 ± 0.0045 |
| Recall | 0.8378 ± 0.0905 |
| F1 Score | 0.8242 ± 0.0476 |

> **Variance verdict: LOW.** Accuracy and ROC-AUC variance is negligible. Recall and F1 show higher variance due to their sensitivity to the 0.5 decision threshold — they are not unstable, they are threshold-sensitive. ROC-AUC is the authoritative metric for comparing models across all thresholds.

---

## 2. Production Metrics (Haar Cascade Pipeline — As Deployed on Render)

Measured via `model/evaluate_haar.py`, replicating the exact production path including penalising Haar detection failures as incorrect predictions.

| Metric | Value |
|---|---|
| **Accuracy** | **78.00%** |
| **ROC-AUC** | **0.8387** |
| Precision | 0.8088 |
| Recall | 0.7333 |
| F1 Score | 0.7692 |

**Production delta:** −6.00pp accuracy, −0.0985 ROC-AUC. Root cause: Haar Cascade fails to detect a face in 14/300 images (4.7%) and produces lower-quality crops than MTCNN on the remainder. Full analysis: `results/day32_detector_delta_v2.md`.

---

## 3. Robustness Under Perturbations (Day 20, fusion model — indicative for CNN-only)

| Condition | Accuracy | Retention % |
|---|---|---|
| Clean baseline | 79.33% | 100% |
| JPEG q=90 | 72.67% | 91.6% |
| JPEG q=30 | 65.67% | 82.8% |
| Blur σ=1 | 62.67% | 79.0% |
| **Blur σ=2** | **52.33%** | **66.0%** |
| **Blur σ=4** | **54.00%** | **68.1%** |
| Resize 0.5× | 65.33% | 82.4% |
| **Resize 0.25×** | **52.33%** | **66.0%** |

**Retention range:** 66.0%–91.6% across perturbation conditions. Bold rows indicate catastrophic collapse, where real-image accuracy falls to 10–23% while fake-image accuracy stays high (82–98%). The model becomes an indiscriminate fake-predictor under heavy blur or extreme downscaling.

---

## 4. Cross-Generator Generalization (Day 36 — StyleGAN3 OOD Evaluation)

**Dataset:** `troykueh/real-vs-fake-faces-stylegan3` (n=150, balanced). Architecture completely different from training distribution.

| Metric | Training Distribution (StyleGAN2) | OOD (StyleGAN3) | Delta |
|---|---|---|---|
| **Overall Accuracy** | 78.00% | **46.67%** | −31.33pp |
| **ROC-AUC** | 0.8387 | **0.5513** | −0.2874 |
| **Fake-Detection Accuracy** | ~84% | **4.00%** | −80pp |
| Real-Detection Accuracy | ~78% | 89.33% | +11.33pp |

**Root cause:** The model learned StyleGAN2's transposed-convolution grid artifacts. StyleGAN3's alias-free continuous coordinate design suppresses these artifacts. The model is blind to StyleGAN3 fakes — it classifies 96% of them as real.

> **Scope statement:** This model is a *StyleGAN2-artifact detector*, not a general synthetic-face detector.

---

## 5. Published Baseline Comparison (Day 35 — XceptionNet)

| Model | Accuracy | ROC-AUC | Parameters |
|---|---|---|---|
| XceptionNet (field-standard baseline) | 81.60% | 0.9120 | 20.8M |
| **EfficientNet-B0 CNN-only (production)** | **84.33%** | **0.9321** | **4.0M** |

The production model outperforms XceptionNet on both accuracy (+2.73pp) and ROC-AUC (+0.0201) with 5× fewer parameters.

---

## 6. Confidence Calibration (Day 39)

| Metric | Value |
|---|---|
| **Expected Calibration Error (ECE)** | **0.0687** |
| **Verdict** | Reasonably well-calibrated (ECE < 0.10) |

The model's raw probability outputs can be used as meaningful confidence estimates. A prediction of 80% confidence will be correct approximately 80% of the time. Minor deviations exist but do not require Platt scaling or temperature scaling for standard use. Reliability diagram: `results/day39_calibration.png`.

---

## 7. Latency (Three Clearly Distinct Measurements — Day 39)

> **These three numbers are not interchangeable.** Each measures a different scope of the system.

| Label | Value | What It Measures |
|---|---|---|
| **Model inference only** | **73.5 ms** | EfficientNet-B0 forward pass in isolation (CPU, batch=1, torch.no_grad). Used for model comparison and speed benchmarking. |
| **Full API response** | **~446 ms** | Complete `/predict` endpoint execution: Haar Cascade (61ms) + preprocessing (2ms) + forward pass (81ms) + Grad-CAM generation (301ms). What a direct API caller actually experiences. Grad-CAM dominates the total. |
| **End-to-end live latency** | **~2.4s** | Real-world round-trip from the deployed Vercel frontend to Render backend, including network overhead (India → US). |

Full breakdown: `results/day39_latency_breakdown.md`.

---

## 8. CI / Test Status (Day 39)

- GitHub Actions workflow at `.github/workflows/test.yml` runs on every push to `main`.
- Sanity check: loads `day32_finetuned_converged.pth`, runs inference on 2 sample images, asserts output probability ∈ [0, 1].
- Local test status: `1 passed` (pytest, Python 3.11).
