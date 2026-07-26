# Day 18: Robustness Perturbation Suite

Today we built the automated robustness perturbation suite to prepare for evaluating how our final production model (`day16_fusion_best.pth`) degrades under real-world image distortions (JPEG re-compression, motion/focus blur, low-resolution downscaling).

## Perturbation Suite Architecture
We generated perturbed versions of all 300 isolated test set images across **9 distinct degradation conditions**:

1. **JPEG Compression** (4 levels):
   - `jpeg_q90`: Mild compression (high quality web upload)
   - `jpeg_q70`: Moderate compression (standard messaging attachment)
   - `jpeg_q50`: Heavy compression (aggressive platform re-compression)
   - `jpeg_q30`: Severe compression (heavy blocking artifacts, low quality)
2. **Gaussian Blur** (3 levels):
   - `blur_s1`: $\sigma = 1.0$ (slight out-of-focus blur)
   - `blur_s2`: $\sigma = 2.0$ (moderate motion/lens blur)
   - `blur_s4`: $\sigma = 4.0$ (heavy blur, high-frequency suppression)
3. **Resize Down-then-Up** (2 levels):
   - `resize_0.5x`: 50% downscaling ($112\times112 \rightarrow 224\times224$ via cubic upsampling)
   - `resize_0.25x`: 25% downscaling ($56\times56 \rightarrow 224\times224$ via cubic upsampling)

**Total Dataset Size**: 9 conditions $\times$ 300 test images = **2,700 perturbed images** stored in `data/perturbed/` (gitignored).

## Multi-Modal Pipeline Integration for Day 19
For the Day 19 robustness evaluation, both branches of our fusion model will process the perturbed images dynamically:
- **CNN Branch**: Image loaded from `data/perturbed/<condition>/<label>/<filename>` $\rightarrow$ PIL image $\rightarrow$ ImageNet normalization $\rightarrow$ EfficientNet-B0 backbone.
- **FFT Branch**: On-the-fly 2D FFT calculation (`extract_radial_profile`) performed directly on the perturbed image to capture how compression/blur distorts the frequency spectrum $\rightarrow$ Z-score normalization $\rightarrow$ FreqBranch MLP.

## Visual Sanity Check
We generated a 10-column visual grid (`results/day18_perturbation_samples.png`) comparing original clean images against all 9 perturbation levels across sample real and fake faces.

**Visual Inspection Results**:
- `jpeg_q30` exhibits sharp 8x8 block boundary artifacts and ringing around facial contours.
- `blur_s4` completely smooths out fine facial textures and hair details.
- `resize_0.25x` shows soft pixelated interpolation artifacts.
- All 9 conditions are visually distinct and correctly formatted at $224\times224$ resolution.
