"""
model/evaluate_xception.py
-----------------------------
Day 35: Evaluate the converged XceptionNet baseline (day34_xception_best.pth)
on the fixed test set.

Outputs:
  results/day35_xception_metrics.md
  results/day35_xception_confusion_matrix.png
  results/day35_xception_roc_curve.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import timm
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, IMAGENET_MEAN, IMAGENET_STD

SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
RESULTS_DIR = ROOT / "results"
CKPT        = ROOT / "model" / "checkpoints" / "day34_xception_best.pth"

# Xception uses 299x299 input size
XCEPTION_SIZE = 299

def get_xception_val_transforms():
    return transforms.Compose([
        transforms.Resize((XCEPTION_SIZE, XCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_model(ckpt_path, device):
    model = timm.create_model("legacy_xception", pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    return model

def evaluate(model, loader, device):
    all_probs, all_preds, all_labels, times = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            t0 = time.time()
            logits = model(images)
            times.append((time.time() - t0) / images.size(0))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return (np.array(all_labels), np.array(all_probs),
            np.array(all_preds), np.mean(times) * 1000)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_ds = DeepfakeFaceDataset(test_df, FACES_DIR, transform=get_xception_val_transforms())
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} images")

    model = load_model(CKPT, device)

    print("Running inference...")
    y_true, y_prob, y_pred, avg_ms = evaluate(model, loader, device)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n--- DAY 35 XCEPTIONNET BASELINE: TEST SET RESULTS ---")
    print(f"Accuracy:        {acc*100:.2f}%")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print(f"Avg Infer. Time: {avg_ms:.2f} ms/image")
    print(f"Confusion Matrix:\n{cm}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real (0)", "Fake (1)"],
                yticklabels=["Real (0)", "Fake (1)"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix (Day 35 XceptionNet)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day35_xception_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve (Day 35 XceptionNet)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day35_xception_roc_curve.png", dpi=150)
    plt.close()

    # Markdown report
    out = RESULTS_DIR / "day35_xception_metrics.md"
    with open(out, "w") as f:
        f.write("# Day 35: XceptionNet Baseline Test Set Metrics\n\n")
        f.write(f"**Checkpoint:** `model/checkpoints/day34_xception_best.pth`\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Accuracy | {acc*100:.2f}% |\n")
        f.write(f"| Precision | {prec:.4f} |\n")
        f.write(f"| Recall | {rec:.4f} |\n")
        f.write(f"| F1 Score | {f1:.4f} |\n")
        f.write(f"| ROC-AUC | {auc:.4f} |\n")
        f.write(f"| Avg Infer. Time | {avg_ms:.2f} ms/image |\n\n")
        f.write(f"## Confusion Matrix\n\n```\n{cm}\n```\n\n*(Rows: True. Cols: Predicted. 0=Real, 1=Fake)*\n")

    print(f"\nMetrics saved: {out}")

if __name__ == "__main__":
    main()
