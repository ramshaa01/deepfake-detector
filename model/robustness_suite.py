"""
model/robustness_suite.py
-------------------------
Day 18: Robustness Perturbation Suite

Generates perturbed copies of test-set face crops under 9 conditions:
- JPEG compression: Quality = 90, 70, 50, 30
- Gaussian blur: Sigma = 1.0, 2.0, 4.0
- Resize-down-then-up: Scale = 0.5x (112x112 -> 224x224), 0.25x (56x56 -> 224x224)

Images are saved to data/perturbed/<condition>/<label>/<filename>.
Generates a visual sanity check grid saved to results/day18_perturbation_samples.png.

Run from project root:
    python model/robustness_suite.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPLITS_DIR    = ROOT / "data" / "splits"
FACES_DIR     = ROOT / "data" / "faces_extracted"
PERTURBED_DIR = ROOT / "data" / "perturbed"
RESULTS_DIR   = ROOT / "results"
SANITY_GRID   = RESULTS_DIR / "day18_perturbation_samples.png"

# ── Perturbation Functions ───────────────────────────────────────────────── #

def apply_jpeg_compression(img_bgr, quality: int) -> np.ndarray:
    """Encodes and decodes an image with JPEG compression at specified quality level."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    if not result:
        raise ValueError(f"JPEG encoding failed at quality {quality}")
    decoded_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return decoded_bgr

def apply_gaussian_blur(img_bgr, sigma: float) -> np.ndarray:
    """Applies Gaussian blur with specified sigma (kernel size auto-calculated)."""
    # Kernel size ksize = 6*sigma + 1 (odd)
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return blurred

def apply_resize(img_bgr, scale: float) -> np.ndarray:
    """Resizes image down by scale factor, then back up to original 224x224."""
    h, w = img_bgr.shape[:2]
    down_w = max(1, int(w * scale))
    down_h = max(1, int(h * scale))
    downsampled = cv2.resize(img_bgr, (down_w, down_h), interpolation=cv2.INTER_AREA)
    upsampled   = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
    return upsampled


# Dictionary of all 9 perturbation conditions: condition_name -> function
CONDITIONS = {
    "jpeg_q90":   lambda img: apply_jpeg_compression(img, 90),
    "jpeg_q70":   lambda img: apply_jpeg_compression(img, 70),
    "jpeg_q50":   lambda img: apply_jpeg_compression(img, 50),
    "jpeg_q30":   lambda img: apply_jpeg_compression(img, 30),
    "blur_s1":    lambda img: apply_gaussian_blur(img, 1.0),
    "blur_s2":    lambda img: apply_gaussian_blur(img, 2.0),
    "blur_s4":    lambda img: apply_gaussian_blur(img, 4.0),
    "resize_0.5x":  lambda img: apply_resize(img, 0.5),
    "resize_0.25x": lambda img: apply_resize(img, 0.25),
}


# ── Batch Generation ─────────────────────────────────────────────────────── #

def generate_perturbed_dataset(test_df):
    print("\n--- Generating Perturbed Dataset Across 9 Conditions ---")
    total_generated = 0

    for cond_name, func in CONDITIONS.items():
        cond_dir = PERTURBED_DIR / cond_name
        (cond_dir / "real").mkdir(parents=True, exist_ok=True)
        (cond_dir / "fake").mkdir(parents=True, exist_ok=True)

        for _, row in test_df.iterrows():
            src_path = FACES_DIR / row["label"] / row["filename"]
            dst_path = cond_dir / row["label"] / row["filename"]

            img_bgr = cv2.imread(str(src_path))
            if img_bgr is None:
                print(f"Error reading {src_path}")
                continue

            perturbed_bgr = func(img_bgr)
            cv2.imwrite(str(dst_path), perturbed_bgr)
            total_generated += 1

        print(f"  Created {len(test_df)} images for condition: {cond_name}")

    print(f"\nSuccessfully generated {total_generated} total perturbed images in {PERTURBED_DIR}")


# ── Visual Sanity Grid ───────────────────────────────────────────────────── #

def generate_sanity_check_grid(test_df):
    print("\n--- Generating Visual Sanity Check Grid ---")
    
    # Pick 3 sample images (mix of real and fake)
    samples = [
        test_df[test_df["label"] == "real"].iloc[0],
        test_df[test_df["label"] == "fake"].iloc[0],
        test_df[test_df["label"] == "real"].iloc[1],
    ]
    
    cond_keys = list(CONDITIONS.keys())  # 9 conditions
    num_cols = len(cond_keys) + 1       # Original + 9 conditions = 10 columns
    num_rows = len(samples)             # 3 sample images
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.2, num_rows * 2.5))

    for row_idx, row_data in enumerate(samples):
        filename = row_data["filename"]
        label    = row_data["label"]
        src_path = FACES_DIR / label / filename
        
        orig_bgr = cv2.imread(str(src_path))
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        
        # Col 0: Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig_rgb)
        if row_idx == 0:
            ax_orig.set_title("Original\n(Clean)", fontsize=10, fontweight="bold")
        ax_orig.set_ylabel(f"{label.upper()}\n{filename[:8]}...", fontsize=9)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        # Cols 1-9: Perturbations
        for col_idx, cond_name in enumerate(cond_keys):
            ax = axes[row_idx, col_idx + 1]
            perturbed_path = PERTURBED_DIR / cond_name / label / filename
            pert_bgr = cv2.imread(str(perturbed_path))
            pert_rgb = cv2.cvtColor(pert_bgr, cv2.COLOR_BGR2RGB)
            
            ax.imshow(pert_rgb)
            if row_idx == 0:
                ax.set_title(cond_name.replace('_', '\n'), fontsize=10, fontweight="bold")
            ax.axis("off")

    fig.suptitle("Day 18 Robustness Suite: Visual Sanity Check Across 9 Perturbation Conditions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(SANITY_GRID, dpi=150, bbox_inches="tight")
    print(f"Saved visual sanity grid to: {SANITY_GRID}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    print(f"Loaded test split with {len(test_df)} samples.")

    generate_perturbed_dataset(test_df)
    generate_sanity_check_grid(test_df)


if __name__ == "__main__":
    main()
