"""
model/calibration_check.py
--------------------------
Calculates Expected Calibration Error (ECE) and plots a reliability diagram.
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_val_test_transforms
from model.evaluate import load_model, evaluate

def compute_calibration(y_true, y_probs, num_bins=10):
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    
    bin_sums = np.bincount(bin_indices, weights=y_probs, minlength=num_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=num_bins)
    bin_total = np.bincount(bin_indices, minlength=num_bins)
    
    nonzero = bin_total != 0
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    
    bin_accs[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    bin_confs[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(bin_accs[nonzero] - bin_confs[nonzero]) * (bin_total[nonzero] / np.sum(bin_total)))
    
    return bin_accs, bin_confs, bin_total, ece

def plot_reliability_diagram(bin_accs, bin_confs, bin_total, ece, save_path):
    plt.figure(figsize=(6, 6))
    
    # Plot perfect calibration
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Plot empirical calibration
    nonzero = bin_total > 0
    plt.plot(bin_confs[nonzero], bin_accs[nonzero], "s-", label=f"Model (ECE = {ece:.4f})")
    
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    plt.legend(loc="lower right")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    ckpt_path = ROOT / "model" / "checkpoints" / "day32_finetuned_converged.pth"
    test_csv = ROOT / "data" / "splits" / "test.csv"
    faces_dir = ROOT / "data" / "faces_extracted"
    results_dir = ROOT / "results"
    
    # 1. Load Data
    test_df = pd.read_csv(test_csv)
    test_ds = DeepfakeFaceDataset(test_df, faces_dir, transform=get_val_test_transforms())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = load_model(ckpt_path, device)
    
    # 3. Run Inference
    y_true, y_probs, y_pred, _ = evaluate(model, test_loader, device)
    
    # 4. Calibration
    bin_accs, bin_confs, bin_total, ece = compute_calibration(y_true, y_probs)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # 5. Plot
    plot_path = results_dir / "day39_calibration.png"
    plot_reliability_diagram(bin_accs, bin_confs, bin_total, ece, plot_path)
    print(f"Saved reliability diagram: {plot_path}")
    
    # 6. Save Report
    report_path = results_dir / "day39_calibration_report.md"
    
    if ece < 0.05:
        interp = "The model is extremely well-calibrated (ECE < 0.05). Its confidence scores accurately reflect the true probability of a face being synthetic. A prediction of 80% confidence will be correct approximately 80% of the time, making the raw probability output highly trustworthy for downstream thresholding."
    elif ece < 0.10:
        interp = "The model is reasonably well-calibrated (ECE < 0.10). Its confidence scores generally reflect true probabilities, with minor deviations. It can be safely used for thresholding with standard caution."
    else:
        interp = "The model is poorly calibrated (ECE > 0.10). It likely exhibits significant overconfidence or underconfidence, meaning raw probability scores should not be interpreted as true likelihoods without applying temperature scaling or Platt scaling."

    with open(report_path, "w") as f:
        f.write("# Day 39: Confidence Calibration Analysis\n\n")
        f.write(f"**Expected Calibration Error (ECE):** {ece:.4f}\n\n")
        f.write("### Interpretation\n")
        f.write(interp + "\n\n")
        f.write("### Reliability Diagram\n")
        f.write("![Reliability Diagram](day39_calibration.png)\n")

    print(f"Saved calibration report: {report_path}")

if __name__ == "__main__":
    main()
