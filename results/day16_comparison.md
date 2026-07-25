# Day 16: Three-Way Model Comparison (Corrected & Investigated)

This document compares the three primary model iterations evaluated on the identical 300-sample test set (150 real, 150 fake).

*Note: Following the Day 16-17 investigation (`results/day16-17_timing_investigation.md`), inference latencies were re-measured under standardized single-image (`batch_size=1`) deployment conditions including real-time FFT computation, and Precision-Recall curves were analyzed to account for threshold shifts.*

## Three-Way Comparison Table

| Metric | Day 7 (Head-Only) | Day 10 (Fine-Tuned CNN) | Day 16 (CNN + FFT Fusion) | Day 10 Tuned (th=0.4249) |
|---|---|---|---|---|
| **Binary Accuracy** | 75.67% | 78.00% | **79.00%** | 77.33% |
| **ROC-AUC** | 0.8249 | 0.8492 | **0.8535** | 0.8492 |
| **PR-AUC (Avg Precision)** | 0.7912 | 0.8196 | **0.8222** | 0.8196 |
| **Precision** | 75.16% | **78.00%** | 75.14% | 73.03% |
| **Recall** | 76.67% | 78.00% | **86.67%** | **86.67%** |
| **F1 Score** | 75.91% | 78.00% | **0.8050** | 0.7927 |
| **Single-Image Latency (Batch=1)** | 195.31 ms | **191.17 ms** | 241.24 ms (+50 ms) | 191.17 ms |

## Key Investigation Findings

1. **Inference Latency Correction**:
   - The original ~33 ms reported for Day 16 omitted the ~50 ms overhead of real-time 2D FFT calculation from raw disk images.
   - When benchmarked under identical single-image upload conditions (`batch_size=1`), the fine-tuned CNN (Day 10) takes **~191 ms**, while the Fusion model (Day 16) takes **~241 ms** due to on-the-fly FFT matrix computation and radial binning.

2. **Precision/Recall & Threshold Analysis**:
   - The apparent large jump in recall (+8.67%) at default threshold `0.50` for Day 16 was predominantly a **threshold artifact**.
   - When Day 10's threshold is tuned to `0.4249` to match Day 16's 86.67% recall, Day 10 achieves 73.03% precision.
   - Comparing both models at matched 86.67% recall shows Fusion delivers a modest +2.11% precision advantage (75.14% vs 73.03%) and a slight threshold-independent PR-AUC gain (**0.8222 vs 0.8196**, +0.0026).

## Conclusion
The fusion model provides a minor ranking gain (+0.0026 PR-AUC) over the CNN-only model, but introduces a **+50 ms latency penalty (+26% slowdown)** per image. In real-world deployment, the CNN-only model (Day 10) provides a more practical balance of speed and accuracy.
