"""
model/run_robustness_eval.py
----------------------------
Day 19: Robustness Evaluation

Evaluates the day16_fusion_best.pth model on the clean test set and 
9 perturbation conditions generated in Day 18.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import get_val_test_transforms
from model.train_fusion import FusionDataset, FUSION_CKPT, FACES_DIR, SPLITS_DIR, FREQ_NPZ
from model.evaluate_fusion import build_model_from_ckpt
from model.frequency_features import extract_radial_profile
from model.robustness_suite import CONDITIONS, PERTURBED_DIR

RESULTS_DIR = ROOT / "results"
JSON_OUT = RESULTS_DIR / "day19_robustness_metrics.json"
MD_OUT = RESULTS_DIR / "day19_robustness_summary.md"


def evaluate_dataset(model, device, dataloader):
    all_labels = []
    all_preds  = []
    all_probs  = []
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, freq, labels in dataloader:
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

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Avg_Inference_Time_ms": np.mean(inference_times)
    }
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model_from_ckpt(FUSION_CKPT, device)

    print("Loading test data...")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    results = {}
    
    # 1. Clean Baseline
    print("\n--- Evaluating Clean Baseline ---")
    npz = np.load(FREQ_NPZ)
    X_test_clean = npz["X_test"]
    clean_ds = FusionDataset(test_df, FACES_DIR, X_test_clean, transform=get_val_test_transforms())
    clean_loader = DataLoader(clean_ds, batch_size=32, shuffle=False, num_workers=0)
    
    clean_metrics = evaluate_dataset(model, device, clean_loader)
    results["clean"] = clean_metrics
    print(f"Clean Accuracy: {clean_metrics['Accuracy']*100:.2f}% | ROC-AUC: {clean_metrics['ROC-AUC']:.4f}")

    # 2. Perturbations
    cond_keys = list(CONDITIONS.keys())
    
    for cond in cond_keys:
        print(f"\n--- Evaluating Condition: {cond} ---")
        cond_dir = PERTURBED_DIR / cond
        
        # Extract features for this condition dynamically
        X_cond = []
        for _, row in test_df.iterrows():
            path = cond_dir / row['label'] / row['filename']
            try:
                profile = extract_radial_profile(path)
            except Exception as e:
                print(f"Error computing FFT for {path}: {e}")
                profile = np.zeros(16)
            X_cond.append(profile)
        X_cond = np.array(X_cond)
        
        # Build dataset and evaluate
        cond_ds = FusionDataset(test_df, cond_dir, X_cond, transform=get_val_test_transforms())
        cond_loader = DataLoader(cond_ds, batch_size=32, shuffle=False, num_workers=0)
        
        cond_metrics = evaluate_dataset(model, device, cond_loader)
        results[cond] = cond_metrics
        print(f"{cond} Accuracy: {cond_metrics['Accuracy']*100:.2f}% | ROC-AUC: {cond_metrics['ROC-AUC']:.4f}")

    # Save to JSON
    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved metrics JSON to {JSON_OUT}")
    
    # Save to Markdown
    with open(MD_OUT, "w") as f:
        f.write("# Day 19: Robustness Evaluation\n\n")
        f.write("Evaluation of the final fusion model (`day16_fusion_best.pth`) across various image perturbations.\n\n")
        f.write("| Condition | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for cond, mets in results.items():
            name = cond if cond != "clean" else "**clean (baseline)**"
            f.write(f"| {name} | {mets['Accuracy']*100:.2f}% | {mets['Precision']:.4f} | {mets['Recall']:.4f} | {mets['F1']:.4f} | {mets['ROC-AUC']:.4f} |\n")
            
    print(f"Saved summary MD to {MD_OUT}")

if __name__ == "__main__":
    main()
