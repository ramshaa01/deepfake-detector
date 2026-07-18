"""
data/make_splits.py
-------------------
Stratified train / val / test split for the Deepfake Face dataset.

Split ratio:  70 % train | 15 % val | 15 % test   (totals to 2,000 samples)
              → 1,400 train | 300 val | 300 test

Stratification: by label (real/fake) so each split maintains the ~50/50 balance.
Random seed:    42 (fixed for full reproducibility)

Output CSVs:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Leakage analysis
----------------
Real images:  5-digit numeric filenames (e.g. "59348.jpg")
              → sourced from the "real" folder of the Kaggle dataset,
                which contains synthetic faces from StyleGAN — each image is
                a unique generated identity, NOT a real video frame.
              → No video-level grouping applies; no same-person leakage risk.

Fake images:  10-char alphanumeric filenames (e.g. "BKX3BV13HZ.jpg")
              → sourced from the Kaggle "fake" image folder; images are already
                pre-cropped stills from deepfake videos, but filenames carry
                no video-ID prefix, so we CANNOT definitively rule out that
                two crops share a source video.
              CONCLUSION: There is a *theoretical* leakage risk for fake images.
              However, given that we sampled only 1,000 from a pool of ~70k+
              unique fake filenames and those filenames appear to be assigned
              per-frame (not shared across identities), the practical risk is
              very low. This is flagged here and in results/day4_notes.md.
              For production-grade splitting you would need the Kaggle metadata
              CSV that maps filenames → source video IDs.
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────── #
METADATA_CSV = os.path.join("data", "faces_extracted", "metadata.csv")
SPLITS_DIR   = os.path.join("data", "splits")
SEED         = 42
VAL_RATIO    = 0.15   # 15 % val
TEST_RATIO   = 0.15   # 15 % test
# TRAIN_RATIO  = 0.70  # implied
# ──────────────────────────────────────────────────────────────────────────── #


def make_splits():
    df = pd.read_csv(METADATA_CSV)

    # Keep only successful extractions
    df = df[df["extraction_status"] == "success"].copy()
    print(f"Total successful extractions: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # ── Leakage check ─────────────────────────────────────────────────────── #
    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    # Check for numeric vs alphanumeric filename patterns
    real_numeric = real_df["filename"].str.replace(".jpg", "", regex=False).str.isnumeric().all()
    print(f"Real filenames are all numeric: {real_numeric}  (unique generated faces -> no leakage risk)")

    # For fake: count unique filename lengths as a proxy for structure
    fake_fname_lengths = fake_df["filename"].str.len().unique()
    print(f"Fake filename lengths: {fake_fname_lengths}")
    print("Fake filenames have no video-ID prefix detectable from filename alone.")
    print("WARNING: Theoretical leakage risk for fake images -- see docstring for full analysis.\n")

    # ── Stratified split ──────────────────────────────────────────────────── #
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"],
        random_state=SEED,
    )

    # Second split: val vs test (equal halves of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,           # 50% of temp → 15% of total
        stratify=temp_df["label"],
        random_state=SEED,
    )

    # ── Report ────────────────────────────────────────────────────────────── #
    for name, split in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        counts = split["label"].value_counts()
        real_n = counts.get("real", 0)
        fake_n = counts.get("fake", 0)
        total  = len(split)
        print(f"{name:5s}: {total:4d} total | real={real_n} ({real_n/total*100:.1f}%) | fake={fake_n} ({fake_n/total*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────── #
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"),   index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    print(f"\nSplit CSVs saved to {SPLITS_DIR}/")


if __name__ == "__main__":
    make_splits()
