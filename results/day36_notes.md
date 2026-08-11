# Day 36 Notes: Cross-Generator Generalization Test

## Data Sources

### Primary OOD Dataset (Used)

| Field | Value |
|---|---|
| **Dataset name** | Real vs Fake Faces (StyleGAN3) |
| **Kaggle URL** | https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3 |
| **License** | CC BY-NC-SA 4.0 |
| **Real source** | FFHQ (same as training real set) |
| **Fake source** | StyleGAN3 ONLY — clean, single-generator labelling |
| **Why chosen** | Isolates generator effect: real-photo source is held constant (FFHQ), so any accuracy drop is attributable entirely to the generator change (StyleGAN2→StyleGAN3), not to a different real-photo domain |
| **Sample size** | 75 real + 75 fake (seed=42 fixed) |

### Rejected Dataset: `chuneeb/deepfake-detection-dataset-2026`

This dataset was initially downloaded but found to be **unusable** for evaluation:
- The dataset labels 3,767 images from `randomuser.me/api/portraits/` as FAKE (StyleGAN3)
- `randomuser.me` serves **real stock photographs** of real people as API avatars, not AI-generated images
- Using this dataset would test nothing meaningful: both classes contain real human photos
- This is a data quality error in the Kaggle dataset, not a flaw in our pipeline
- **Decision:** discarded; switched to `troykueh/real-vs-fake-faces-stylegan3` which has verified StyleGAN3 outputs

## Methodology

- **Face extraction:** Haar Cascade (identical to Render production pipeline — `evaluate_haar.py` convention)
- **Model:** `day32_finetuned_converged.pth` — the deployed production model
- **Haar detection failures:** 5/150 (3.3%) — penalised as wrong predictions (same methodology as Day 31/32 Haar delta measurement)
- **Label convention:** 0=REAL, 1=FAKE (consistent with all prior evaluation)
- **Random seed:** 42 (fixed for reproducibility)