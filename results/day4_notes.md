# Day 4 Notes: PyTorch Dataset + Stratified Train/Val/Test Split

## Summary
Built the full PyTorch data pipeline — Dataset class, transforms, stratified splits, DataLoaders, and sanity checks. All components verified working.

## Files Created
| File | Purpose |
|---|---|
| `data/dataset.py` | `DeepfakeFaceDataset` PyTorch Dataset class + transform pipelines |
| `data/make_splits.py` | Stratified 70/15/15 split generator, saves CSVs |
| `data/verify_dataloaders.py` | Wires up DataLoaders and asserts batch shapes/dtypes |
| `data/sanity_check.py` | Class balance report, sample image grids, leakage check |
| `data/splits/train.csv` | 1,400 samples |
| `data/splits/val.csv` | 300 samples |
| `data/splits/test.csv` | 300 samples |
| `results/day4_*_sample_grid.png` | Sample image grids per split |

## Label Convention
**Used consistently throughout this project:**
- `0` = REAL (authentic face)
- `1` = FAKE (manipulated / deepfake face)

## Split Sizes and Class Balance

| Split | Total | REAL (0) | FAKE (1) | Balance |
|---|---|---|---|---|
| **Train** | 1,400 | 700 (50.0%) | 700 (50.0%) | Perfect |
| **Val** | 300 | 150 (50.0%) | 150 (50.0%) | Perfect |
| **Test** | 300 | 150 (50.0%) | 150 (50.0%) | Perfect |

Split ratio: **70/15/15** — standard for this dataset size. Stratified by label with `random_state=42`.

## DataLoader Verification

All three DataLoaders confirmed working:
- **Batch shape:** `(32, 3, 224, 224)` float32 ✅
- **Label dtype:** `torch.int64` ✅
- **Label values:** strictly in `{0, 1}` ✅
- **Train batch balance:** real=16, fake=16 (shuffled, approximately balanced) ✅

## Transform Pipelines

**Training:** Resize(224) → RandomHorizontalFlip(p=0.5) → RandomRotation(±10°) → ColorJitter → Normalize(ImageNet)

**Val/Test:** Resize(224) → Normalize(ImageNet) — no augmentation

> NOTE: Aggressive blur, JPEG compression, or frequency augmentations were explicitly excluded from training augmentation because they would destroy the subtle forensic artifacts (GAN fingerprints, blending boundaries) that EfficientNet-B0 needs to learn to detect.

## Leakage Check Results
- **Real filenames:** All 5-digit numeric IDs (StyleGAN-generated unique faces). Zero overlap across any split pair. **No leakage.**
- **Fake filenames:** All 10-char alphanumeric IDs. Zero filename overlap across any split pair.
- **Filename-level result:** `[PASS]` — zero overlap in all 6 split-pair combinations.
- **Video-level leakage: CLOSED (Day 5)** — Investigation of the Kaggle source CSV metadata (`train.csv`, `test.csv`, `valid.csv`) confirmed:
  - REAL images come from **NVIDIA FFHQ** (Flickr Faces HQ) — StyleGAN-synthesised individual portraits. No videos, no sequences.
  - FAKE images come from **"1 Million Fake Faces"** — StyleGAN2-synthesised individual portraits. No videos, no sequences.
  - Neither dataset contains video frames. The Day 4 caveat was based on the assumption this might be a video-frame dataset (like DFDC). It is not.
  - **No group-aware re-splitting needed.** Current splits are fully valid.

## Sanity Check Grids
Sample image grids (4×4) for each split saved to:
- `results/day4_train_sample_grid.png`
- `results/day4_val_sample_grid.png`
- `results/day4_test_sample_grid.png`
