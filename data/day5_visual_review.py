"""
data/day5_visual_review.py
--------------------------
Day 5 visual quality pass: 20 real + 20 fake random samples reviewed for:
  - Mislabeling
  - Bad crops / extraction artifacts
  - Corrupted or unusable images

Run from project root:
    python data/day5_visual_review.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FACES_DIR   = os.path.join("data", "faces_extracted")
RESULTS_DIR = "results"
SEED        = 99   # Different from split seed so we get fresh samples
N_PER_CLASS = 20


def main():
    meta = pd.read_csv(os.path.join(FACES_DIR, "metadata.csv"))
    meta = meta[meta["extraction_status"] == "success"].reset_index(drop=True)

    real_sample = meta[meta["label"] == "real"].sample(N_PER_CLASS, random_state=SEED)
    fake_sample = meta[meta["label"] == "fake"].sample(N_PER_CLASS, random_state=SEED)
    samples = pd.concat([real_sample, fake_sample]).reset_index(drop=True)

    LABEL_COLOR = {"real": "#4CAF50", "fake": "#F44336"}
    cols = 8
    rows = 5   # 40 images / 8 cols = 5 rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = axes.flatten()

    issues = []

    for i, (ax, (_, row)) in enumerate(zip(axes, samples.iterrows())):
        img_path = os.path.join(FACES_DIR, row["label"], row["filename"])
        color = LABEL_COLOR[row["label"]]
        issue = None

        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)

            # Check for black/blank images (mean pixel < 5)
            if arr.mean() < 5:
                issue = "BLANK"

            # Check for near-uniform images (very low std = featureless)
            elif arr.std() < 8:
                issue = "FEATURELESS"

            # Check for unexpected size (should all be 224x224)
            elif img.size != (224, 224):
                issue = f"SIZE:{img.size}"

            ax.imshow(arr)

        except Exception as e:
            issue = "MISSING"
            ax.text(0.5, 0.5, "NOT FOUND", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=8)

        title = row["label"].upper()
        if issue:
            title += f"\n[{issue}]"
            issues.append({"filename": row["filename"], "label": row["label"], "issue": issue})
            color = "#FF9800"  # Orange for issues

        ax.set_title(title, color=color, fontsize=7, fontweight="bold", pad=2)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Day 5 Visual Quality Review — 20 REAL + 20 FAKE (random sample)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "day5_visual_review.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Grid saved to {out_path}")

    print(f"\nIssues found: {len(issues)}")
    if issues:
        for iss in issues:
            print(f"  [{iss['issue']}] {iss['label']}/{iss['filename']}")
    else:
        print("  None — all 40 samples passed visual checks.")

    return issues


if __name__ == "__main__":
    main()
