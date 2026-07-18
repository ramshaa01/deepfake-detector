"""
data/sanity_check.py
--------------------
Sanity check script for the Day 4 data pipeline.

Confirms:
  1. Class balance across train/val/test splits
  2. Saves a grid of sample images per split to results/
  3. Leakage check: reports on the potential same-person/video leakage risk

Run from project root:
    python data/sanity_check.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

SPLITS_DIR  = os.path.join("data", "splits")
FACES_DIR   = os.path.join("data", "faces_extracted")
RESULTS_DIR = "results"
LABEL_MAP   = {0: "REAL", 1: "FAKE"}
LABEL_COLOR = {"real": "#4CAF50", "fake": "#F44336"}


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(SPLITS_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    return df[df["extraction_status"] == "success"].copy()


def class_balance_report(df: pd.DataFrame, name: str):
    counts = df["label"].value_counts()
    real_n = counts.get("real", 0)
    fake_n = counts.get("fake", 0)
    total  = len(df)
    print(f"  {name:5s}: {total:4d} total | real={real_n} ({real_n/total*100:.1f}%) | fake={fake_n} ({fake_n/total*100:.1f}%)")


def save_sample_grid(df: pd.DataFrame, name: str, n_samples: int = 16):
    """Save a 4x4 grid of sample images with label overlays."""
    rows_real = df[df["label"] == "real"].sample(min(n_samples // 2, len(df[df["label"] == "real"])), random_state=42)
    rows_fake = df[df["label"] == "fake"].sample(min(n_samples // 2, len(df[df["label"] == "fake"])), random_state=42)
    sample_df = pd.concat([rows_real, rows_fake]).sample(frac=1, random_state=42).reset_index(drop=True)

    cols = 4
    rows = max(1, len(sample_df) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i, (ax, (_, row)) in enumerate(zip(axes, sample_df.iterrows())):
        img_path = os.path.join(FACES_DIR, row["label"], row["filename"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(np.array(img))
        except Exception:
            ax.text(0.5, 0.5, "NOT FOUND", ha="center", va="center", transform=ax.transAxes, color="red")

        color = LABEL_COLOR[row["label"]]
        ax.set_title(row["label"].upper(), color=color, fontsize=9, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused axes
    for ax in axes[len(sample_df):]:
        ax.set_visible(False)

    fig.suptitle(f"{name.upper()} split — sample face crops", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"day4_{name}_sample_grid.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved grid -> {out_path}")


def leakage_check(train_df, val_df, test_df):
    print("\nLeakage Check")
    print("  Real filenames (numeric IDs — StyleGAN-generated unique faces):")
    all_real = set(train_df[train_df["label"]=="real"]["filename"]) | \
               set(val_df[val_df["label"]=="real"]["filename"]) | \
               set(test_df[test_df["label"]=="real"]["filename"])
    train_real = set(train_df[train_df["label"]=="real"]["filename"])
    val_real   = set(val_df[val_df["label"]=="real"]["filename"])
    test_real  = set(test_df[test_df["label"]=="real"]["filename"])
    tv_overlap = train_real & val_real
    tt_overlap = train_real & test_real
    vt_overlap = val_real   & test_real
    print(f"    train/val overlap:  {len(tv_overlap)} files")
    print(f"    train/test overlap: {len(tt_overlap)} files")
    print(f"    val/test overlap:   {len(vt_overlap)} files")

    print("  Fake filenames (10-char alphanumeric):")
    train_fake = set(train_df[train_df["label"]=="fake"]["filename"])
    val_fake   = set(val_df[val_df["label"]=="fake"]["filename"])
    test_fake  = set(test_df[test_df["label"]=="fake"]["filename"])
    tv_f = train_fake & val_fake
    tt_f = train_fake & test_fake
    vt_f = val_fake   & test_fake
    print(f"    train/val overlap:  {len(tv_f)} files")
    print(f"    train/test overlap: {len(tt_f)} files")
    print(f"    val/test overlap:   {len(vt_f)} files")

    if len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0 \
       and len(tv_f) == 0 and len(tt_f) == 0 and len(vt_f) == 0:
        print("  [PASS] Zero filename overlap across all splits -- no filename-level leakage.")
    else:
        print("  [WARN] Filename overlap detected -- investigate further!")
    print("  NOTE: Video-level leakage for fake images cannot be fully ruled out")
    print("        without Kaggle source video metadata (see make_splits.py docstring).")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    print("=== Class Balance Report ===")
    class_balance_report(train_df, "TRAIN")
    class_balance_report(val_df,   "VAL")
    class_balance_report(test_df,  "TEST")

    print("\n=== Sample Image Grids ===")
    save_sample_grid(train_df, "train")
    save_sample_grid(val_df,   "val")
    save_sample_grid(test_df,  "test")

    leakage_check(train_df, val_df, test_df)
    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
