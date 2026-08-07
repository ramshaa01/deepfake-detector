"""
model/evaluate_day32_cnn.py
-----------------------------
Day 32: Evaluate the converged CNN checkpoint (day32_finetuned_converged.pth)
on the test set. Mirrors Day 10's evaluation exactly, so results are directly
comparable.

Outputs:
  results/day32_converged_cnn_metrics.md
  results/day32_cnn_confusion_matrix.png
  results/day32_cnn_roc_curve.png

Run from project root:
    python model/evaluate_day32_cnn.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_val_test_transforms

SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
RESULTS_DIR = ROOT / "results"
CKPT        = ROOT / "model" / "checkpoints" / "day32_finetuned_converged.pth"

# Day 10 official numbers (for direct comparison)
DAY10 = dict(acc=0.7800, prec=0.7800, rec=0.7800, f1=0.7800, auc=0.8492)


def load_model(ckpt_path, device):
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
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
    device = torch.device("cpu")
    RESULTS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_ds = DeepfakeFaceDataset(test_df, FACES_DIR, transform=get_val_test_transforms())
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

    print("\n--- DAY 32 CONVERGED CNN: TEST SET RESULTS ---")
    print(f"Accuracy:        {acc*100:.2f}%  (Day 10: {DAY10['acc']*100:.2f}%  delta: {(acc-DAY10['acc'])*100:+.2f}%)")
    print(f"Precision:       {prec:.4f}  (Day 10: {DAY10['prec']:.4f}  delta: {prec-DAY10['prec']:+.4f})")
    print(f"Recall:          {rec:.4f}  (Day 10: {DAY10['rec']:.4f}  delta: {rec-DAY10['rec']:+.4f})")
    print(f"F1 Score:        {f1:.4f}  (Day 10: {DAY10['f1']:.4f}  delta: {f1-DAY10['f1']:+.4f})")
    print(f"ROC-AUC:         {auc:.4f}  (Day 10: {DAY10['auc']:.4f}  delta: {auc-DAY10['auc']:+.4f})")
    print(f"Avg Infer. Time: {avg_ms:.2f} ms/image")
    print(f"Confusion Matrix:\n{cm}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real (0)", "Fake (1)"],
                yticklabels=["Real (0)", "Fake (1)"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix (Day 32 Converged CNN)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day32_cnn_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve (Day 32 Converged CNN)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day32_cnn_roc_curve.png", dpi=150)
    plt.close()

    # Markdown report
    delta_acc  = (acc  - DAY10["acc"])  * 100
    delta_auc  = auc   - DAY10["auc"]
    delta_f1   = f1    - DAY10["f1"]
    verdict = "improved" if delta_acc > 0.5 else ("marginally improved" if delta_acc > 0 else "did NOT improve")

    out = RESULTS_DIR / "day32_converged_cnn_metrics.md"
    with open(out, "w") as f:
        f.write("# Day 32: Converged CNN Test Set Metrics\n\n")
        f.write(f"**Checkpoint:** `model/checkpoints/day32_finetuned_converged.pth`\n")
        f.write(f"**Comparison baseline:** Day 10 (day8_finetuned_best.pth, interrupted at epoch 2)\n\n")
        f.write("## Direct Comparison: Day 32 (Converged) vs Day 10 (Interrupted)\n\n")
        f.write("| Metric | Day 10 (Interrupted) | Day 32 (Converged) | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Accuracy | {DAY10['acc']*100:.2f}% | {acc*100:.2f}% | {delta_acc:+.2f}% |\n")
        f.write(f"| Precision | {DAY10['prec']:.4f} | {prec:.4f} | {prec-DAY10['prec']:+.4f} |\n")
        f.write(f"| Recall | {DAY10['rec']:.4f} | {rec:.4f} | {rec-DAY10['rec']:+.4f} |\n")
        f.write(f"| F1 Score | {DAY10['f1']:.4f} | {f1:.4f} | {delta_f1:+.4f} |\n")
        f.write(f"| ROC-AUC | {DAY10['auc']:.4f} | {auc:.4f} | {delta_auc:+.4f} |\n")
        f.write(f"| Avg Infer. Time | — | {avg_ms:.2f} ms | — |\n\n")
        f.write(f"## Interpretation\n\n")
        f.write(f"Full convergence **{verdict}** on CNN-only accuracy relative to the interrupted Day 8 checkpoint.\n")
        f.write(f"Accuracy delta: {delta_acc:+.2f}%. ROC-AUC delta: {delta_auc:+.4f}. F1 delta: {delta_f1:+.4f}.\n\n")
        f.write(f"Note: Both numbers are CNN-only baselines. The fusion model evaluation follows separately.\n\n")
        f.write(f"## Confusion Matrix\n\n```\n{cm}\n```\n\n*(Rows: True. Cols: Predicted. 0=Real, 1=Fake)*\n")

    print(f"\nMetrics saved: {out}")


if __name__ == "__main__":
    main()
