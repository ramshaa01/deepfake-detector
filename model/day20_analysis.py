import sys
import time
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import get_val_test_transforms
from model.train_fusion import FusionDataset, FUSION_CKPT, FACES_DIR, SPLITS_DIR, FREQ_NPZ
from model.evaluate_fusion import build_model_from_ckpt
from model.frequency_features import extract_radial_profile
from model.robustness_suite import CONDITIONS, PERTURBED_DIR

RESULTS_DIR = ROOT / "results"
CSV_OUT = RESULTS_DIR / "day20_per_class_breakdown.csv"
MD_OUT = RESULTS_DIR / "day20_robustness_table.md"
CHART_OUT = RESULTS_DIR / "day20_robustness_chart.png"
INVESTIGATION_OUT = RESULTS_DIR / "day20_cliff_investigation.json"

def get_predictions(model, device, dataloader):
    all_labels = []
    all_preds  = []
    all_probs  = []
    all_cnn_feats = []
    all_freq_feats = []
    
    model.eval()
    with torch.no_grad():
        for images, freq, labels in dataloader:
            images = images.to(device)
            freq   = freq.to(device)

            # Manual forward pass to get intermediate features
            cnn_feat = model.cnn(images)
            freq_out = model.freq_branch(freq)
            combined = torch.cat([cnn_feat, freq_out], dim=1)
            logits = model.head(combined)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            
            all_cnn_feats.extend(cnn_feat.cpu().numpy())
            all_freq_feats.extend(freq_out.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_cnn_feats), np.array(all_freq_feats)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(FUSION_CKPT, device)

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    cond_keys = ["clean"] + list(CONDITIONS.keys())
    
    results = {}
    predictions_dict = {}

    for cond in cond_keys:
        print(f"Evaluating {cond}...")
        if cond == "clean":
            cond_dir = FACES_DIR
            npz = np.load(FREQ_NPZ)
            X_cond = npz["X_test"]
        else:
            cond_dir = PERTURBED_DIR / cond
            X_cond = []
            for _, row in test_df.iterrows():
                path = cond_dir / row['label'] / row['filename']
                try:
                    profile = extract_radial_profile(path)
                except:
                    profile = np.zeros(16)
                X_cond.append(profile)
            X_cond = np.array(X_cond)

        ds = FusionDataset(test_df, cond_dir, X_cond, transform=get_val_test_transforms())
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        y_true, y_pred, y_prob, cnn_feats, freq_feats = get_predictions(model, device, loader)
        
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        real_mask = (y_true == 0)
        fake_mask = (y_true == 1)
        
        real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask])
        fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask])
        
        results[cond] = {
            "accuracy": acc,
            "roc_auc": auc,
            "real_accuracy": real_acc,
            "fake_accuracy": fake_acc,
            "real_n": np.sum(real_mask),
            "fake_n": np.sum(fake_mask)
        }
        
        predictions_dict[cond] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "cnn_feats": cnn_feats,
            "freq_feats": freq_feats
        }

    # 1. Save per-class breakdown CSV
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "real_accuracy", "fake_accuracy", "real_n", "fake_n"])
        for cond in cond_keys:
            r = results[cond]
            writer.writerow([cond, f"{r['real_accuracy']:.4f}", f"{r['fake_accuracy']:.4f}", r['real_n'], r['fake_n']])

    # 2. Save robustness table MD
    clean_acc = results["clean"]["accuracy"]
    with open(MD_OUT, "w") as f:
        f.write("# Day 20: Robustness Evaluation Table\n\n")
        f.write("| Condition | Accuracy | ROC-AUC | Retention % | Real Accuracy | Fake Accuracy |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cond in cond_keys:
            r = results[cond]
            retention = (r["accuracy"] / clean_acc) * 100
            f.write(f"| {cond} | {r['accuracy']*100:.2f}% | {r['roc_auc']:.4f} | {retention:.1f}% | {r['real_accuracy']*100:.2f}% | {r['fake_accuracy']*100:.2f}% |\n")
            
    # 3. Plot chart
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # JPEG
    jpeg_conds = ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"]
    jpeg_x = [100, 90, 70, 50, 30]
    jpeg_y = [results[c]["accuracy"] * 100 for c in jpeg_conds]
    axs[0].plot(jpeg_x, jpeg_y, marker='o')
    axs[0].set_xlim(100, 20) # decreasing quality
    axs[0].set_title("JPEG Compression")
    axs[0].set_xlabel("Quality (100 = Clean)")
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].set_ylim(40, 85)
    axs[0].grid(True)
    
    # Blur
    blur_conds = ["clean", "blur_s1", "blur_s2", "blur_s4"]
    blur_x = [0, 1, 2, 4]
    blur_y = [results[c]["accuracy"] * 100 for c in blur_conds]
    axs[1].plot(blur_x, blur_y, marker='o', color='orange')
    axs[1].set_title("Gaussian Blur")
    axs[1].set_xlabel("Sigma (0 = Clean)")
    axs[1].set_ylim(40, 85)
    axs[1].grid(True)
    
    # Resize
    resize_conds = ["clean", "resize_0.5x", "resize_0.25x"]
    resize_x = [1.0, 0.5, 0.25]
    resize_y = [results[c]["accuracy"] * 100 for c in resize_conds]
    axs[2].plot(resize_x, resize_y, marker='o', color='green')
    axs[2].set_xlim(1.0, 0.0) # decreasing scale
    axs[2].set_title("Resizing (Down/Up)")
    axs[2].set_xlabel("Scale (1.0 = Clean)")
    axs[2].set_ylim(40, 85)
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(CHART_OUT, dpi=150)
    
    # 4. Investigate blur_s1 vs blur_s2 cliff
    s1_data = predictions_dict["blur_s1"]
    s2_data = predictions_dict["blur_s2"]
    
    # Find images that were correct in blur_s1 but flipped in blur_s2
    correct_s1 = (s1_data["y_pred"] == s1_data["y_true"])
    correct_s2 = (s2_data["y_pred"] == s2_data["y_true"])
    flipped_idx = np.where(correct_s1 & ~correct_s2)[0]
    
    investigation = {
        "cliff_analysis": {
            "num_flipped_images": int(len(flipped_idx)),
            "s1_avg_prob_on_flipped": float(np.mean(s1_data["y_prob"][flipped_idx])),
            "s2_avg_prob_on_flipped": float(np.mean(s2_data["y_prob"][flipped_idx])),
            "s1_avg_cnn_feat_norm": float(np.mean(np.linalg.norm(s1_data["cnn_feats"], axis=1))),
            "s2_avg_cnn_feat_norm": float(np.mean(np.linalg.norm(s2_data["cnn_feats"], axis=1))),
            "s1_avg_freq_feat_norm": float(np.mean(np.linalg.norm(s1_data["freq_feats"], axis=1))),
            "s2_avg_freq_feat_norm": float(np.mean(np.linalg.norm(s2_data["freq_feats"], axis=1)))
        }
    }
    
    with open(INVESTIGATION_OUT, "w") as f:
        json.dump(investigation, f, indent=4)
        
    print(f"Saved all Day 20 outputs to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
