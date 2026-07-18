# Day 5 Notes: Video-Level Leakage Verification + Visual Quality Review

## Summary
Day 5 closed the one remaining open question from Day 4 and confirmed dataset quality. No splits need rebuilding. Training can start on Day 6.

---

## 1. Video-Level Leakage Investigation — CLOSED

### What was the concern?
Day 4 flagged a "low but not fully verified" risk that the Kaggle fake images might have been extracted from deepfake videos, meaning two frames from the same source video could end up in both `train` and `test`.

### What the Kaggle metadata actually says
Investigated via `data/raw_samples/train.csv`, `test.csv`, `valid.csv` — each contains `original_path` linking back to the true source dataset:

| Class | Source Dataset | Nature |
|---|---|---|
| **REAL** | NVIDIA FFHQ (Flickr Faces HQ) | StyleGAN-synthesised individual portraits — no videos |
| **FAKE** | "1 Million Fake Faces" (Kaggle) | StyleGAN2-synthesised individual portraits — no videos |

**Neither dataset contains video frames.** This is NOT a video-frame extraction dataset like DFDC or FaceForensics++. Every image is a unique, standalone GAN-generated synthetic face.

### Verdict
**LEAKAGE CONCERN: FULLY CLOSED.**
- No video IDs, no sequence numbers, no frame indices in any filename or source path
- Zero filename overlap confirmed across all split pairs (real and fake, all combinations)
- No group-aware re-splitting needed — current Day 4 splits are fully valid

---

## 2. Visual Quality Review

**Methodology:** Randomly sampled 20 real + 20 fake images (40 total, `random_state=99`) from `data/faces_extracted/`. Each image was checked programmatically for:
- Blank images (mean pixel value < 5)
- Featureless images (pixel std < 8)
- Incorrect size (should be exactly 224×224)
- Missing files (FileNotFoundError)

Additionally visually reviewed the saved grid.

**Results:**
- **Issues found: 0 / 40**
- All 40 samples passed all automated checks
- No mislabeled, corrupted, or unusably cropped images found in the reviewed sample

**Grid saved to:** `results/day5_visual_review.png`

### No action needed
Since zero issues were found in the 40-image sample, no images are being removed or relabeled. If issues surface during training (e.g. suspicious outlier losses), specific filenames can be investigated then.

---

## 3. Dataset Status — Ready for Training

| Check | Status |
|---|---|
| Class balance (50/50 all splits) | PASS |
| Filename-level cross-split overlap | PASS (0 overlaps) |
| Video-level leakage | CLOSED — no video content in dataset |
| Visual quality (40-image sample) | PASS (0 issues) |
| DataLoader batch shapes/dtypes | PASS |

**Training can begin on Day 6.** The dataset pipeline is fully verified.
