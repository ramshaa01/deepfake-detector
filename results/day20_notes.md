# Day 20: Robustness Analysis & Root-Cause Investigation

## 1. Per-Class Breakdown Findings
Evaluating the accuracy separately for Real vs. Fake images reveals exactly *how* the model fails under perturbation:
- Under moderate degradation (e.g., `jpeg_q70`, `resize_0.5x`), the model is relatively stable, though Real accuracy drops faster than Fake accuracy.
- Under heavy degradation (`blur_s2`, `blur_s4`, `resize_0.25x`), the model **collapses to predicting almost everything as Fake**. For instance, under `blur_s4`, Fake accuracy is 98.00% while Real accuracy is just 10.00%. The model's failure mode is a severe false-positive spiral.

## 2. Robustness Retention Table

| Condition | Accuracy | ROC-AUC | Retention % | Real Accuracy | Fake Accuracy |
|---|---|---|---|---|---|
| clean | 79.33% | 0.8544 | 100.0% | 72.00% | 86.67% |
| jpeg_q90 | 72.67% | 0.8268 | 91.6% | 58.67% | 86.67% |
| jpeg_q70 | 68.67% | 0.7727 | 86.6% | 50.00% | 87.33% |
| jpeg_q50 | 65.33% | 0.7358 | 82.4% | 48.67% | 82.00% |
| jpeg_q30 | 65.67% | 0.7112 | 82.8% | 56.00% | 75.33% |
| blur_s1 | 62.67% | 0.6466 | 79.0% | 56.67% | 68.67% |
| blur_s2 | 52.33% | 0.5363 | 66.0% | 22.67% | 82.00% |
| blur_s4 | 54.00% | 0.5192 | 68.1% | 10.00% | 98.00% |
| resize_0.5x | 65.33% | 0.7128 | 82.4% | 46.00% | 84.67% |
| resize_0.25x | 52.33% | 0.5244 | 66.0% | 13.33% | 91.33% |

## 3. The `blur_s1` -> `blur_s2` Cliff Investigation
There is a sharp accuracy cliff dropping from 62.67% at `blur_s1` down to 52.33% at `blur_s2`. Investigating the 73 images that flipped between these two conditions:
- **Probability Shift**: The average prediction probability on these borderline images shifted from `0.41` (Real) in `s1` to `0.63` (Fake) in `s2`. 
- **Root Cause**: The severe blurring destroys the high-frequency cues (e.g., sharp skin textures, fine edges) that the model relies on to classify an image as a genuine photograph. The fusion model, inherently sensitive to pixel-level and spectral frequencies (amplified by the FFT branch), interprets the absence of these high-frequency "real" features as indicative of a fake image. Consequently, any heavily smoothed or downscaled image gets defaulted to a "Fake" prediction.

## 4. One-Line Summary
**Robust to moderate JPEG compression typical of social media re-uploads, but fragile to heavy blur and severe downscaling, under which it collapses to predicting almost all faces as AI-generated.**
