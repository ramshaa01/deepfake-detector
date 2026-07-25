# Day 16-17 Investigation: Inference Timing & Precision-Recall Analysis

Prior to advancing to Day 17, we conducted a rigorous investigation into two critical inconsistencies observed in Day 16's evaluation:
1. **Inference Latency Anomaly**: The fusion model (CNN + FFT + MLP) was originally reported as faster (~33 ms) than the CNN-only model (~58–142 ms), despite performing more total operations.
2. **Precision/Recall Tradeoff**: The default threshold for the fusion model showed a large recall jump (+8.67%) accompanied by a precision drop (-2.42%), raising the question of whether this was a true discriminative gain or merely a threshold shift artifact.

---

## Investigation 1: Inference Timing Methodology & Correction

### Root Cause Analysis
- **Evaluation Flaw in Day 16**: In `evaluate_fusion.py`, frequency features were loaded directly from the pre-computed `data/frequency_features.npz` file in memory. The timing calculation completely omitted raw image disk loading and real-time 2D FFT computation. Furthermore, batch processing (`batch_size=32`) parallelized tensor operations on CPU, artificially reducing the per-sample time calculation.
- **Evaluation Flaw in Day 7/10**: Per-image times were derived from batch-divided execution times under varying system/CPU load across different script invocations, creating artificial variance (~142 ms vs ~58 ms) for identical `EfficientNet-B0` forward passes.

### Corrected Methodology
We implemented a standardized, real-world benchmark (`model/benchmark_investigation.py`):
- **Single-Image Batch Size (`batch_size = 1`)**: Simulates real-world API deployment where single images are uploaded.
- **Warmup Exclusions**: 10 warmup runs excluded per model to avoid cold-start/JIT overhead.
- **Controlled System State**: All 3 models evaluated sequentially in the exact same Python process environment across 100 test images.
- **Full End-to-End Pipeline for Fusion (Day 16)**: Includes raw image disk load, PIL conversion, tensor transform, **on-the-fly 2D FFT extraction** (`extract_radial_profile`), z-score feature normalization, and the full CNN + MLP forward pass.

### Corrected Results (Single-Image Batch=1 Latency on CPU)

| Model Iteration | Pipeline Components | Corrected Latency (Mean ± Std) | Real-World Impact |
|---|---|---|---|
| **Day 7 (Head-Only CNN)** | Image Load + Transform + CNN Forward | **195.31 ms ± 23.62 ms** | Baseline CNN speed |
| **Day 10 (Fine-Tuned CNN)** | Image Load + Transform + CNN Forward | **191.17 ms ± 19.36 ms** | Identical to Day 7 (same architecture) |
| **Day 16 (CNN + FFT Fusion)** | Image Load + Transform + **On-the-Fly FFT** + Fusion Forward | **241.24 ms ± 28.38 ms** | **+50.07 ms overhead (+26.2%)** |

**Conclusion**: Under fair, deployment-realistic conditions, the fusion model is **~50 ms slower per image** than the CNN-only model due to the CPU computation of the 2D FFT matrix and radial binning (~46 ms) plus the extra MLP layer forward pass (~4 ms).

---

## Investigation 2: Precision-Recall Curve & Threshold Shift Analysis

### Hypothesis
The fusion model's reported recall increase (78.00% -> 86.67%) and precision drop (78.00% -> 75.58%) at default threshold (0.50) could simply reflect a downward shift in decision threshold calibration rather than genuine ranking capability improvements.

### Threshold-Independent Evaluation (ROC-AUC & PR-AUC)
To evaluate ranking quality independent of decision thresholds, we computed the Precision-Recall Area Under Curve (PR-AUC / Average Precision) alongside ROC-AUC:

- **Day 10 (CNN-Only)**: ROC-AUC = **0.8492** | PR-AUC (Average Precision) = **0.8196**
- **Day 16 (Fusion)**:   ROC-AUC = **0.8535** | PR-AUC (Average Precision) = **0.8222** (+0.0026)

### Matched Operating Point Analysis
We calibrated Day 10's decision threshold to achieve the **exact same Recall (86.67%)** as Day 16's default output:

| Model & Threshold | Recall | Precision | Accuracy |
|---|---|---|---|
| **Day 10 (CNN-only, default threshold `0.5000`)** | 78.00% | 78.00% | 78.00% |
| **Day 10 (CNN-only, tuned threshold `0.4249`)** | **86.67%** | **73.03%** | **77.33%** |
| **Day 16 (Fusion, default threshold `0.5000`)** | **86.67%** | **75.14%** | **79.00%** |

### Key Findings
1. **The Recall Jump Was Predominantly a Threshold Shift Artifact**: Adjusting Day 10's decision threshold from `0.5000` to `0.4249` achieves the exact same 86.67% recall on synthetic faces without any model changes.
2. **Fusion Provides a Genuine but Marginal Improvement (+0.0026 PR-AUC)**: At the matched 86.67% recall operating point, Day 16 Fusion maintains slightly better Precision (75.14% vs 73.03%, +2.11%) and Accuracy (79.00% vs 77.33%, +1.67%) than threshold-tuned Day 10.
3. **PR Curve Alignment**: Visual inspection of the PR curve (`results/day16_pr_curve_comparison.png`) demonstrates that the Day 10 and Day 16 precision-recall curves track each other almost identically across almost all recall levels.

---

## Overall Summary & Project Framing
- **Inference Speed**: Fusion adds **+50 ms latency penalty** (+26% slowdown on CPU) for real-time applications.
- **Classification Performance**: Fusion yields only a **+0.0026 PR-AUC / +0.0043 ROC-AUC** improvement. The dramatic +8.67% recall boost originally reported was primarily a threshold artifact.
- **Final Decision**: The FFT radial-energy fusion experiment is documented as a valuable tested hypothesis that confirms modern GANs (StyleGAN2) require deep spatial features rather than global frequency statistics.
