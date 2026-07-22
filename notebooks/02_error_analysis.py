"""
notebooks/02_error_analysis.py
------------------------------
Day 12: Error Analysis on the Fine-Tuned Model

Runs inference on the test set using day8_finetuned_best.pth.
Identifies False Positives and False Negatives, records confidence scores,
calculates FPR and FNR, and plots the hardest errors.

Run from project root:
    python notebooks/02_error_analysis.py
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import timm

# ── Path setup ────────────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
SPLITS_DIR   = ROOT / "data" / "splits"
FACES_DIR    = ROOT / "data" / "faces_extracted"
CKPT_PATH    = ROOT / "model" / "checkpoints" / "day8_finetuned_best.pth"
RESULTS_DIR  = ROOT / "results"

# ── Model ─────────────────────────────────────────────────────────────────── #
def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

# ── Main ──────────────────────────────────────────────────────────────────── #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_ds = DeepfakeFaceDataset(test_df, FACES_DIR, transform=get_val_test_transforms())
    # shuffle=False guarantees order matches test_df exactly
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    model = load_model(CKPT_PATH, device)
    
    # Run Inference
    print("Running inference over test set for error analysis...")
    all_probs = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            
    test_df["prob_fake"] = all_probs
    test_df["pred"] = (test_df["prob_fake"] > 0.5).astype(int)
    
    # True labels: 0=real, 1=fake
    test_df["true_label"] = test_df["label"].map({"real": 0, "fake": 1})
    
    # Identify errors
    errors_df = test_df[test_df["pred"] != test_df["true_label"]].copy()
    
    # False Positives: True=Real(0), Pred=Fake(1)
    fp_df = errors_df[(errors_df["true_label"] == 0) & (errors_df["pred"] == 1)].copy()
    # False Negatives: True=Fake(1), Pred=Real(0)
    fn_df = errors_df[(errors_df["true_label"] == 1) & (errors_df["pred"] == 0)].copy()
    
    total_real = (test_df["true_label"] == 0).sum()
    total_fake = (test_df["true_label"] == 1).sum()
    
    fpr = len(fp_df) / total_real if total_real > 0 else 0
    fnr = len(fn_df) / total_fake if total_fake > 0 else 0
    
    print("\n--- ERROR RATES ---")
    print(f"Total Test Samples: {len(test_df)} ({total_real} Real, {total_fake} Fake)")
    print(f"False Positives (Real flagged as Fake): {len(fp_df)}  -> FPR: {fpr:.2%}")
    print(f"False Negatives (Fake flagged as Real): {len(fn_df)}  -> FNR: {fnr:.2%}")
    
    # Calculate confidence: for FP, confidence is prob_fake (closer to 1.0).
    # For FN, confidence in "Real" is 1.0 - prob_fake (closer to 0.0 means closer to 1.0 in Real prob).
    fp_df["confidence"] = fp_df["prob_fake"]
    fn_df["confidence"] = 1.0 - fn_df["prob_fake"]
    
    fp_hard = fp_df.sort_values(by="confidence", ascending=False).head(6)
    fn_hard = fn_df.sort_values(by="confidence", ascending=False).head(6)
    
    hardest_errors = pd.concat([fp_hard, fn_hard])
    
    # Plotting the hardest errors
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(hardest_errors.iterrows()):
        ax = axes[i]
        img_path = FACES_DIR / row["label"] / row["filename"]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(np.array(img))
        except Exception as e:
            ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
            
        true_str = row["label"].upper()
        pred_str = "FAKE" if row["pred"] == 1 else "REAL"
        color = "red"
        
        title = f"True: {true_str}\nPred: {pred_str}\nConf: {row['confidence']:.1%}"
        ax.set_title(title, color=color, fontsize=10, fontweight="bold")
        ax.axis("off")
        
    for j in range(len(hardest_errors), len(axes)):
        axes[j].axis("off")
        
    fig.suptitle("Hardest Errors: Most Confident False Positives (Top) & False Negatives (Bottom)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "day12_hardest_errors.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved hardest errors grid to: {plot_path}")
    
    # Dump paths of hardest errors for qualitative review markdown
    print("\nHardest False Positives (Real flagged as Fake):")
    for _, row in fp_hard.iterrows():
        print(f"  - {row['filename']} (Conf: {row['confidence']:.1%})")
        
    print("\nHardest False Negatives (Fake flagged as Real):")
    for _, row in fn_hard.iterrows():
        print(f"  - {row['filename']} (Conf: {row['confidence']:.1%})")

if __name__ == "__main__":
    main()
