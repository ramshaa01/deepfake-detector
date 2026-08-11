# Day 36: Cross-Generator Generalization Test

## Setup

| Item | Value |
|---|---|
| **Production model** | `day32_finetuned_converged.pth` (EfficientNet-B0, CNN-only) |
| **Training distribution** | Real: FFHQ; Fake: StyleGAN2 |
| **OOD real source** | FFHQ (same real-photo source; isolates generator effect) |
| **OOD fake source** | StyleGAN3 (different generator from StyleGAN2 training) |
| **Dataset** | `troykueh/real-vs-fake-faces-stylegan3` (Kaggle, CC BY-NC-SA 4.0) |
| **Sample size** | 75 real + 75 fake = 150 total (seed=42) |
| **Face extraction** | Haar Cascade (identical to Render production pipeline) |

## Results: In-Distribution vs Out-of-Distribution

| Metric | In-Distribution (Day 32, Haar, StyleGAN2) | OOD (Day 36, Haar, StyleGAN3) | Delta |
|---|---|---|---|
| **Overall Accuracy** | **78.00%** | **46.67%** | **-31.33pp** |
| Precision | 0.8088 | 0.2727 | -0.5361 |
| Recall | 0.7333 | 0.0400 | -0.6933 |
| F1 Score | 0.7692 | 0.0698 | -0.6994 |
| ROC-AUC | 0.8387 | 0.5513 | -0.2874 |
| Haar failures | 4.7% (14/300) | 3.3% (5/150) | — |

### Per-Class Accuracy (OOD)

| Class | Accuracy |
|---|---|
| Real images (FFHQ, OOD source) | 89.33% |
| Fake images (StyleGAN3, OOD generator) | 4.00% |

### Confusion Matrix (OOD, 0=Real, 1=Fake)

```
[[67  8]
 [72  3]]
```

*(Rows: True class. Cols: Predicted class.)*

## Interpretation

The accuracy drop from in-distribution to OOD is **severe** — the model is near-random on StyleGAN3 fakes.

Delta: **-31.33 percentage points** (ROC-AUC: -0.2874).

Per-class breakdown:
- Real image accuracy: **89.33%** — measures whether the model still correctly identifies real photos (Unsplash/FFHQ real-world photos vs FFHQ training real photos).
- Fake image accuracy: **4.00%** — the critical number: this directly measures whether StyleGAN2-trained detection transfers to StyleGAN3.

### What This Means

The model was trained exclusively on StyleGAN2-generated fakes paired with FFHQ real faces.
StyleGAN2 and StyleGAN3 differ significantly in their generation mechanism:
- **StyleGAN2** uses transposed convolutions → characteristic grid-pattern spectral artefacts at specific frequencies
- **StyleGAN3** uses alias-free synthesis with continuous coordinates → different (or suppressed) frequency artefacts

If fake_acc drops sharply, the model has learned StyleGAN2-specific spectral signatures rather than general GAN-detection features. This is expected and scientifically important.

**Scope conclusion:** This result explicitly quantifies the model's generalization boundary.
The detection capability is scoped to faces from the StyleGAN2 training distribution.
Claims of 'AI-generated face detection' should be qualified as 'StyleGAN2-generated face detection' in research contexts.