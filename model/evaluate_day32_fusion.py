"""
model/evaluate_day32_fusion.py
--------------------------------
Day 32: Evaluate the newly-trained fusion model (day32_fusion_converged.pth)
on the test set. Includes:
  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Operating-point analysis vs the day32 CNN-only model
  - Threshold artifact check (mirrors Day 16-17 rigor)
  - Side-by-side table vs Day 16 official numbers

Outputs:
  results/day32_fusion_converged_metrics.md
  results/day32_fusion_confusion_matrix.png
  results/day32_fusion_roc_curve.png
  results/day32_fusion_pr_curve.png

Run from project root:
    python model/evaluate_day32_fusion.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix, roc_curve, precision_recall_curve)
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import get_val_test_transforms

SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
FREQ_NPZ    = ROOT / "data" / "frequency_features.npz"
RESULTS_DIR = ROOT / "results"
CKPT_DIR    = ROOT / "model" / "checkpoints"
FUSION_CKPT = CKPT_DIR / "day32_fusion_converged.pth"
CNN_CKPT    = CKPT_DIR / "day32_finetuned_converged.pth"  # for op-point comparison

# Day 16 official numbers (for comparison)
DAY16 = dict(acc=0.7933, prec=0.7558, rec=0.8667, f1=0.8075, auc=0.8544, pr_auc=0.8222)


# ── FusionDataset (same as training) ──────────────────────────────────────── #
class FusionDataset(Dataset):
    LABEL_MAP = {"real": 0, "fake": 1}
    def __init__(self, df, faces_dir, freq_features, transform=None):
        self.df = df.reset_index(drop=True)
        self.faces_dir = Path(faces_dir)
        self.transform = transform
        freq = freq_features.astype(np.float32)
        mean = freq.mean(axis=0); std = freq.std(axis=0) + 1e-8
        self.freq_norm = (freq - mean) / std
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.faces_dir / row["label"] / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        freq  = torch.tensor(self.freq_norm[idx], dtype=torch.float32)
        label = torch.tensor(self.LABEL_MAP[row["label"]], dtype=torch.long)
        return image, freq, label


class FreqBranch(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, cnn_backbone):
        super().__init__()
        self.cnn = cnn_backbone; self.freq_branch = FreqBranch()
        self.head = nn.Linear(1280 + 16, 1)
    def forward(self, images, freq_feat):
        return self.head(torch.cat([self.cnn(images), self.freq_branch(freq_feat)], dim=1))


def build_fusion_from_ckpt(ckpt_path, device):
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.classifier = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False
    model = FusionModel(backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def build_fusion_from_fusion_ckpt(ckpt_path, cnn_ckpt_path, device):
    """Load backbone from CNN ckpt, then load fusion weights on top."""
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)

    cnn_ckpt = torch.load(cnn_ckpt_path, map_location=device)
    backbone.load_state_dict(cnn_ckpt["model_state_dict"])
    backbone.classifier = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False

    model = FusionModel(backbone).to(device)
    fusion_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(fusion_ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded fusion checkpoint: epoch={fusion_ckpt['epoch']}, val_acc={fusion_ckpt['val_acc']:.4f}")
    return model


def evaluate_fusion(model, loader, device):
    all_probs, all_preds, all_labels, times = [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, freq, labels in loader:
            images = images.to(device); freq = freq.to(device)
            t0 = time.time()
            logits = model(images, freq)
            times.append((time.time() - t0) / images.size(0))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs), np.array(all_preds), np.mean(times)*1000


def operating_point_analysis(y_true, y_prob_fusion, y_prob_cnn, target_recalls=(0.80, 0.85, 0.867, 0.90)):
    """At each target recall, find threshold and compare precision between models."""
    rows = []
    for target_rec in target_recalls:
        for name, probs in [("Fusion", y_prob_fusion), ("CNN-only", y_prob_cnn)]:
            prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, probs)
            # Find threshold closest to target recall
            idx = np.argmin(np.abs(rec_arr - target_rec))
            rows.append({
                "target_recall": target_rec,
                "model": name,
                "actual_recall": rec_arr[idx],
                "precision": prec_arr[idx],
                "threshold": thresh_arr[idx] if idx < len(thresh_arr) else 1.0,
            })
    return rows


def main():
    device = torch.device("cpu")
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load freq features
    npz = np.load(FREQ_NPZ)
    X_test = npz["X_test"]
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_ds = FusionDataset(test_df, FACES_DIR, X_test, transform=get_val_test_transforms())
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} images")

    # Load fusion model
    fusion_model = build_fusion_from_fusion_ckpt(FUSION_CKPT, CNN_CKPT, device)

    print("Running fusion inference...")
    y_true, y_prob_fusion, y_pred, avg_ms = evaluate_fusion(fusion_model, loader, device)

    # Core metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob_fusion)
    pr_auc = average_precision_score(y_true, y_prob_fusion)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n--- DAY 32 CONVERGED FUSION: TEST SET RESULTS ---")
    print(f"Accuracy:  {acc*100:.2f}%  (Day 16: {DAY16['acc']*100:.2f}%  delta: {(acc-DAY16['acc'])*100:+.2f}%)")
    print(f"Precision: {prec:.4f}  (Day 16: {DAY16['prec']:.4f}  delta: {prec-DAY16['prec']:+.4f})")
    print(f"Recall:    {rec:.4f}  (Day 16: {DAY16['rec']:.4f}  delta: {rec-DAY16['rec']:+.4f})")
    print(f"F1:        {f1:.4f}  (Day 16: {DAY16['f1']:.4f}  delta: {f1-DAY16['f1']:+.4f})")
    print(f"ROC-AUC:   {auc:.4f}  (Day 16: {DAY16['auc']:.4f}  delta: {auc-DAY16['auc']:+.4f})")
    print(f"PR-AUC:    {pr_auc:.4f}  (Day 16: {DAY16['pr_auc']:.4f}  delta: {pr_auc-DAY16['pr_auc']:+.4f})")
    print(f"Infer. ms: {avg_ms:.2f}")
    print(f"Confusion Matrix:\n{cm}")

    # Threshold artifact check (check if 0.5 is a reasonable threshold)
    thresholds = np.arange(0.3, 0.8, 0.05)
    print("\n--- THRESHOLD ARTIFACT CHECK ---")
    print(f"{'Threshold':>10} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
    for t in thresholds:
        yp = (y_prob_fusion > t).astype(int)
        print(f"{t:>10.2f} | {accuracy_score(y_true, yp)*100:>5.1f}% | "
              f"{precision_score(y_true, yp, zero_division=0):>6.4f} | "
              f"{recall_score(y_true, yp, zero_division=0):>6.4f} | "
              f"{f1_score(y_true, yp, zero_division=0):>6.4f}")

    # Also load CNN-only model for operating-point comparison
    from model.evaluate_day32_cnn import load_model as load_cnn
    from data.dataset import DeepfakeFaceDataset
    cnn_model = load_cnn(CNN_CKPT, device)
    cnn_model.eval()
    cnn_test_ds = DeepfakeFaceDataset(test_df, FACES_DIR, transform=get_val_test_transforms())
    cnn_loader  = DataLoader(cnn_test_ds, batch_size=32, shuffle=False, num_workers=0)
    all_cnn_probs = []
    with torch.no_grad():
        for images, labels in cnn_loader:
            logits = cnn_model(images.to(device))
            all_cnn_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    y_prob_cnn = np.array(all_cnn_probs)

    op_rows = operating_point_analysis(y_true, y_prob_fusion, y_prob_cnn)
    print("\n--- OPERATING POINT ANALYSIS ---")
    print(f"{'Target Recall':>13} | {'Model':>10} | {'Actual Recall':>13} | {'Precision':>9} | {'Threshold':>9}")
    for r in op_rows:
        print(f"{r['target_recall']:>13.3f} | {r['model']:>10} | "
              f"{r['actual_recall']:>13.4f} | {r['precision']:>9.4f} | {r['threshold']:>9.4f}")

    # Plots
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real (0)", "Fake (1)"],
                yticklabels=["Real (0)", "Fake (1)"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix (Day 32 Converged Fusion)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day32_fusion_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob_fusion)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Fusion (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve (Day 32 Converged Fusion)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day32_fusion_roc_curve.png", dpi=150)
    plt.close()

    # PR curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob_fusion)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_arr, prec_arr, color="green", lw=2, label=f"Fusion (PR-AUC={pr_auc:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR Curve (Day 32 Converged Fusion)")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "day32_fusion_pr_curve.png", dpi=150)
    plt.close()

    # Markdown report
    op_table = "| Target Recall | Model | Actual Recall | Precision | Threshold |\n|---|---|---|---|---|\n"
    for r in op_rows:
        op_table += f"| {r['target_recall']:.3f} | {r['model']} | {r['actual_recall']:.4f} | {r['precision']:.4f} | {r['threshold']:.4f} |\n"

    out = RESULTS_DIR / "day32_fusion_converged_metrics.md"
    with open(out, "w") as f:
        f.write("# Day 32: Converged Fusion Model Test Set Metrics\n\n")
        f.write(f"**Checkpoint:** `model/checkpoints/day32_fusion_converged.pth`\n")
        f.write(f"**CNN Backbone:** `day32_finetuned_converged.pth` (trained to full convergence)\n")
        f.write(f"**Comparison:** Day 16 (`day16_fusion_best.pth`, based on interrupted day8 backbone)\n\n")
        f.write("## Direct Comparison: Day 32 (Converged) vs Day 16 (Interrupted backbone)\n\n")
        f.write("| Metric | Day 16 (Official) | Day 32 (Converged) | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Accuracy | {DAY16['acc']*100:.2f}% | {acc*100:.2f}% | {(acc-DAY16['acc'])*100:+.2f}% |\n")
        f.write(f"| Precision | {DAY16['prec']:.4f} | {prec:.4f} | {prec-DAY16['prec']:+.4f} |\n")
        f.write(f"| Recall | {DAY16['rec']:.4f} | {rec:.4f} | {rec-DAY16['rec']:+.4f} |\n")
        f.write(f"| F1 Score | {DAY16['f1']:.4f} | {f1:.4f} | {f1-DAY16['f1']:+.4f} |\n")
        f.write(f"| ROC-AUC | {DAY16['auc']:.4f} | {auc:.4f} | {auc-DAY16['auc']:+.4f} |\n")
        f.write(f"| PR-AUC | {DAY16['pr_auc']:.4f} | {pr_auc:.4f} | {pr_auc-DAY16['pr_auc']:+.4f} |\n")
        f.write(f"| Avg Infer. Time | ~241 ms | {avg_ms:.2f} ms | — |\n\n")
        f.write(f"## Operating Point Analysis (Fusion vs CNN-only, matched recall)\n\n")
        f.write(op_table)
        f.write(f"\n## Confusion Matrix\n\n```\n{cm}\n```\n")
        f.write(f"\n*(Rows: True. Cols: Predicted. 0=Real, 1=Fake)*\n")

    print(f"\nMetrics saved: {out}")
    return acc, prec, rec, f1, auc, pr_auc, avg_ms


if __name__ == "__main__":
    main()
