# Final Metrics Summary & Project Wrap-Up

## Final Model Architecture
The production model is a **Fusion Architecture (CNN + FFT)**:
- **Spatial Branch:** EfficientNet-B0 (pre-trained on ImageNet, fine-tuned on face crops).
- **Frequency Branch:** 16-bin 2D FFT radial profile extractor, capturing spectral GAN artefacts.
- **Fusion:** A Multi-Layer Perceptron (MLP) combines the 1280-dim CNN embedding with the 16-dim normalized frequency features to produce the final classification logit.
- **Interpretability:** Hook-based Grad-CAM extracts attention heatmaps from the CNN's final convolutional layer.

## Final Test-Set Metrics
Evaluated on a strictly held-out, balanced dataset of 300 face crops (150 Real from FFHQ, 150 Fake from StyleGAN2).

| Metric | Value |
|---|---|
| **Accuracy** | 79.33% |
| **ROC-AUC** | 0.8544 |
| **Precision** | 0.7558 |
| **Recall** | 0.8667 |
| **F1 Score** | 0.8075 |

## Robustness Retention
The model was tested against 9 perturbation conditions (JPEG compression, Gaussian blur, resizing).

- **Best-Case Retention (JPEG q=90):** 91.6% retention relative to baseline (Accuracy drops from 79.33% to 72.67%).
- **Worst-Case Collapse (Blur σ=4):** 68.1% retention relative to baseline, but Real image accuracy drops catastrophically to **10.00%**, rendering the model unusable (predicts almost all images as Fake).
- **Severe Downscaling (Resize 0.25×):** 66.0% retention, with Real image accuracy dropping to **13.33%**.

## Inference Latency
- **Backend-Only Compute (Render CPU):** ~241–256 ms per image (measured locally and via deployed API logs).
- **Live End-to-End (Vercel UI → Render API):** ~2.4 seconds (measured client-side, India → US East). 
  > *Note: The ~2.15s gap between backend and end-to-end latency is entirely network-dominated transfer time, not compute.*

## Documented Limitations
1. **Eyeglasses False-Positive Bias:** Grad-CAM analysis confirmed the model learned a spurious correlation, associating eyeglasses with synthetic faces due to a training set distribution mismatch. Real people wearing glasses have an elevated false-positive rate.
2. **Catastrophic Blur/Downscale Collapse:** The model relies on high-frequency spectral GAN artefacts. When images are heavily blurred (σ≥2) or downscaled (0.25×), these artefacts are destroyed, causing the model to default to predicting "Fake" (Real accuracy collapses to 10–23%).
3. **Cold-Start Delay:** The live backend is deployed on Render's free tier, which sleeps after 15 minutes of inactivity. The first request after an idle period takes 30–90 seconds to process while the container wakes up.
