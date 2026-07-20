"""
model/evaluate.py
-----------------
Day 7: Official Test-Set Evaluation (Head-Only Baseline)

Loads the trained day6_head_only_best.pth model, runs it over the test set,
and computes standard classification metrics (Accuracy, Precision, Recall,
F1, ROC-AUC). Also generates a confusion matrix and ROC curve plot.

Records average per-image inference time for performance tracking.

Run from project root:
    python model/evaluate.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import timm

# ── Path setup ────────────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
SPLITS_DIR   = ROOT / "data" / "splits"
FACES_DIR    = ROOT / "data" / "faces_extracted"
CKPT_PATH    = ROOT / "model" / "checkpoints" / "day6_head_only_best.pth"
RESULTS_DIR  = ROOT / "results"

# ── Model ─────────────────────────────────────────────────────────────────── #
def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    
    # Reconstruct the custom head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    
    # Load weights
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

# ── Evaluation ────────────────────────────────────────────────────────────── #
def evaluate(model, dataloader, device):
    all_preds_probs = []
    all_preds_binary = []
    all_labels = []
    
    inference_times = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Measure inference time per batch
            t0 = time.time()
            logits = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.time()
            
            # Store time scaled to per-image
            inference_times.append((t1 - t0) / images.size(0))
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = labels.cpu().numpy()
            
            all_preds_probs.extend(probs.flatten())
            all_preds_binary.extend(preds.flatten())
            all_labels.extend(labels.flatten())
            
    avg_inference_time_ms = np.mean(inference_times) * 1000
    return np.array(all_labels), np.array(all_preds_probs), np.array(all_preds_binary), avg_inference_time_ms


# ── Plotting ──────────────────────────────────────────────────────────────── #
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real (0)", "Fake (1)"],
                yticklabels=["Real (0)", "Fake (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Head-Only Baseline)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Head-Only Baseline)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────── #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_ds = DeepfakeFaceDataset(test_df, FACES_DIR, transform=get_val_test_transforms())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test Set Size: {len(test_ds)} images")
    
    # 2. Load Model
    if not CKPT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        print("Please ensure Day 6 training finished successfully.")
        sys.exit(1)
        
    model = load_model(CKPT_PATH, device)
    
    # 3. Run Inference
    print("Running inference over test set...")
    y_true, y_probs, y_pred, avg_inf_ms = evaluate(model, test_loader, device)
    
    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    
    print("\n--- BASELINE METRICS ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\nAvg Inference Time: {avg_inf_ms:.2f} ms / image")
    
    # 5. Generate Plots
    cm_path = RESULTS_DIR / "day7_confusion_matrix.png"
    roc_path = RESULTS_DIR / "day7_roc_curve.png"
    
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Saved confusion matrix: {cm_path}")
    
    plot_roc_curve(y_true, y_probs, roc_path)
    print(f"Saved ROC curve: {roc_path}")
    
    # 6. Save Markdown Report
    report_path = RESULTS_DIR / "day7_baseline_metrics.md"
    with open(report_path, "w") as f:
        f.write("# Day 7: Official Head-Only Baseline Metrics\n\n")
        f.write("These metrics represent the performance of the **EfficientNet-B0 (Head-Only)** model on the isolated test set. ")
        f.write("This is the official baseline to beat during full fine-tuning (Day 8+).\n\n")
        f.write("## Metrics (Test Set)\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {prec:.4f}\n")
        f.write(f"- **Recall:** {rec:.4f}\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n")
        f.write(f"- **ROC-AUC:** {auc:.4f}\n\n")
        f.write("## Performance\n")
        f.write(f"- **Avg Inference Time:** {avg_inf_ms:.2f} ms per image (on {device.type})\n\n")
        f.write("## Plots\n")
        f.write("- `results/day7_confusion_matrix.png`\n")
        f.write("- `results/day7_roc_curve.png`\n")
        
    print(f"Saved metrics report: {report_path}")

if __name__ == "__main__":
    main()
