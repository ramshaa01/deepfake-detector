# Day 15: FFT-based Frequency-Domain Features

Today we explored using frequency-domain signals as a standalone, interpretable feature for deepfake detection. While some literature suggests that GANs leave distinct high-frequency checkerboard artifacts (spectral peaks) due to upsampling operations, we wanted to test if this held true for our specific dataset (FFHQ vs StyleGAN2) using a simple, classical feature.

## Feature Extraction Method
- **Process**: We converted each face crop to grayscale, resized to 256x256, and computed the 2D Fast Fourier Transform (FFT). We shifted the zero-frequency component to the center and computed the log-magnitude spectrum.
- **Feature Vector**: To create a robust, rotation-invariant feature, we calculated the **radial energy distribution**. We divided the spectrum into 16 concentric circular bins (from low frequency at the center to high frequency at the edge) and averaged the log-magnitude within each bin. This resulted in a 1D feature vector of length 16.

## Visual Profile Comparison
We plotted the average 16-bin radial profile across 30 random Real and 30 random Fake images (`results/day15_fft_profiles.png`).

**Finding**: The average radial energy profiles for Real and Fake images are virtually identical. Both follow the standard natural-image $1/f$ falloff curve. There are no distinct spectral peaks, bumps, or anomalies in the high-frequency bins for the StyleGAN2 fakes. The "Real" curve is microscopically higher in the mid-frequencies, but the distributions overlap almost completely.

## Standalone Classifier Performance
We extracted this 16-dimensional feature for the entire train/val/test splits and trained a simple Logistic Regression classifier exclusively on this data (no CNN involved).

**Results on Test Set**:
- **Accuracy**: 54.33%
- **ROC-AUC**: 0.5856

*(For comparison: our baseline head-only CNN achieved 75.67% / 0.8249, and our fine-tuned CNN achieved 78.00% / 0.8492).*

## Conclusion
This simple 1D radial frequency signal is **extremely weak**, barely performing above random guessing (50%). 

Why didn't it work? 
1. **StyleGAN2 Improvements**: StyleGAN2 specifically introduced architectural changes (like removing progressive growing and changing upsampling methods) to reduce the classic GAN checkerboard artifacts that this radial FFT method is designed to catch.
2. **Crop Misalignment**: We extracted faces using MTCNN. Global FFT artifacts are highly sensitive to grid alignment; cropping and scaling faces likely destroyed the uniform pixel-grid phase alignment needed for a global frequency peak to emerge clearly.
3. **Loss of Spatial Locality**: Averaging into a 1D radial profile destroys all spatial and directional frequency information, which might still contain subtle cues.

Ultimately, this proves that for modern, high-quality generators like StyleGAN2 in wildly cropped datasets, simple global frequency statistics are insufficient. We must rely on the deep spatial feature extractors (like our EfficientNet-B0 CNN) to find localized, complex artifacts.
