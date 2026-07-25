"""
model/evaluate_fusion.py
------------------------
Day 16: Evaluate the fusion model on the test set.

Loads day16_fusion_best.pth and computes:
  Accuracy, Precision, Recall, F1, ROC-AUC, Inference Time
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import get_val_test_transforms
from model.train_fusion import FusionDataset, FusionModel, FUSION_CKPT, FACES_DIR, SPLITS_DIR, FREQ_NPZ


def build_model_from_ckpt(ckpt_path, device):
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)  # match ckpt key structure

    # Load weights into full EfficientNet first (to match saved state dict)
    # The saved model IS a FusionModel, so we need to build that
    backbone.classifier = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False

    model = FusionModel(backbone).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model_from_ckpt(FUSION_CKPT, device)

    print("Loading test data...")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    npz = np.load(FREQ_NPZ)
    X_test = npz["X_test"]

    from torch.utils.data import DataLoader
    test_ds     = FusionDataset(test_df, FACES_DIR, X_test, transform=get_val_test_transforms())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_labels = []
    all_preds  = []
    all_probs  = []
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, freq, labels in test_loader:
            images = images.to(device)
            freq   = freq.to(device)

            t0 = time.perf_counter()
            logits = model(images, freq)
            t1 = time.perf_counter()

            batch_time = (t1 - t0) / images.size(0) * 1000  # ms per image
            inference_times.append(batch_time)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred)
    rec   = recall_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob)
    cm    = confusion_matrix(y_true, y_pred)
    avg_t = np.mean(inference_times)

    print("\n--- FUSION MODEL TEST SET RESULTS ---")
    print(f"Accuracy:        {acc*100:.2f}%")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print(f"Avg Infer. Time: {avg_t:.2f} ms/image")
    print(f"Confusion Matrix:\n{cm}")

    out_path = ROOT / "results" / "day16_fusion_metrics.md"
    with open(out_path, "w") as f:
        f.write("# Day 16: Fusion Model — Test Set Metrics\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Accuracy | {acc*100:.2f}% |\n")
        f.write(f"| Precision | {prec:.4f} |\n")
        f.write(f"| Recall | {rec:.4f} |\n")
        f.write(f"| F1 Score | {f1:.4f} |\n")
        f.write(f"| ROC-AUC | {auc:.4f} |\n")
        f.write(f"| Avg Inference Time | {avg_t:.2f} ms/image |\n")
        f.write(f"\n### Confusion Matrix\n\n```\n{cm}\n```\n")
        f.write(f"\n*(Rows: True, Cols: Predicted. 0=Real, 1=Fake)*\n")
    print(f"\nMetrics saved to {out_path}")

    # Return for use in comparison script
    return acc, prec, rec, f1, auc, avg_t


if __name__ == "__main__":
    main()
