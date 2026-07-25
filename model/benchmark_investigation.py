"""
model/benchmark_investigation.py
--------------------------------
Day 16-17 Investigation:
1. Corrected, fair inference timing (single image batch_size=1, on-the-fly FFT for fusion, warmup runs).
2. PR-curve, PR-AUC (Average Precision), and matched operating point analysis (Day 10 vs Day 16).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import get_val_test_transforms
from model.frequency_features import extract_radial_profile
from model.train_fusion import FusionModel

SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
RESULTS_DIR = ROOT / "results"
FREQ_NPZ    = ROOT / "data" / "frequency_features.npz"

DAY6_CKPT   = ROOT / "model" / "checkpoints" / "day6_head_only_best.pth"
DAY8_CKPT   = ROOT / "model" / "checkpoints" / "day8_finetuned_best.pth"
DAY16_CKPT  = ROOT / "model" / "checkpoints" / "day16_fusion_best.pth"

device = torch.device("cpu")


# ── Model Builders ────────────────────────────────────────────────────────── #
def load_cnn_model(ckpt_path: Path) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def load_fusion_model(ckpt_path: Path) -> nn.Module:
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)
    backbone.classifier = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False
    model = FusionModel(backbone).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Part 1: Inference Timing Benchmark ────────────────────────────────────── #
def run_timing_benchmark(test_df, transform):
    print("\n=======================================================")
    print(" PART 1: RIGOROUS INFERENCE TIMING BENCHMARK (Batch=1) ")
    print("=======================================================")
    
    day7_model  = load_cnn_model(DAY6_CKPT)
    day10_model = load_cnn_model(DAY8_CKPT)
    day16_model = load_fusion_model(DAY16_CKPT)
    
    # Load 100 sample image paths for timing
    samples = test_df.sample(n=100, random_state=42).reset_index(drop=True)
    
    # Pre-load normalizer constants for fusion FFT
    npz = np.load(FREQ_NPZ)
    mean_fft = npz["X_train"].mean(axis=0)
    std_fft  = npz["X_train"].std(axis=0) + 1e-8

    # Warmup runs (10 iterations)
    dummy_img = Image.new("RGB", (224, 224))
    dummy_tensor = transform(dummy_img).unsqueeze(0).to(device)
    dummy_freq = torch.zeros((1, 16), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            day7_model(dummy_tensor)
            day10_model(dummy_tensor)
            day16_model(dummy_tensor, dummy_freq)

    # 1. Day 7 Timing (Head-Only CNN: Image Load -> Transform -> Forward)
    times_day7 = []
    with torch.no_grad():
        for _, row in samples.iterrows():
            img_path = FACES_DIR / row["label"] / row["filename"]
            t0 = time.perf_counter()
            img = Image.open(img_path).convert("RGB")
            t_tensor = transform(img).unsqueeze(0).to(device)
            _ = torch.sigmoid(day7_model(t_tensor))
            t1 = time.perf_counter()
            times_day7.append((t1 - t0) * 1000)

    # 2. Day 10 Timing (Fine-Tuned CNN: Image Load -> Transform -> Forward)
    times_day10 = []
    with torch.no_grad():
        for _, row in samples.iterrows():
            img_path = FACES_DIR / row["label"] / row["filename"]
            t0 = time.perf_counter()
            img = Image.open(img_path).convert("RGB")
            t_tensor = transform(img).unsqueeze(0).to(device)
            _ = torch.sigmoid(day10_model(t_tensor))
            t1 = time.perf_counter()
            times_day10.append((t1 - t0) * 1000)

    # 3. Day 16 Timing (Fusion: Image Load -> Transform + On-the-fly FFT -> Forward)
    times_day16 = []
    with torch.no_grad():
        for _, row in samples.iterrows():
            img_path = FACES_DIR / row["label"] / row["filename"]
            t0 = time.perf_counter()
            # Image transform
            img = Image.open(img_path).convert("RGB")
            t_tensor = transform(img).unsqueeze(0).to(device)
            # Real-time FFT feature extraction
            fft_raw = extract_radial_profile(img_path)
            fft_norm = (fft_raw - mean_fft) / std_fft
            freq_tensor = torch.tensor(fft_norm, dtype=torch.float32).unsqueeze(0).to(device)
            # Forward pass
            _ = torch.sigmoid(day16_model(t_tensor, freq_tensor))
            t1 = time.perf_counter()
            times_day16.append((t1 - t0) * 1000)

    t7_mean,  t7_std  = np.mean(times_day7),  np.std(times_day7)
    t10_mean, t10_std = np.mean(times_day10), np.std(times_day10)
    t16_mean, t16_std = np.mean(times_day16), np.std(times_day16)

    print(f"Day 7  (Head-Only CNN):  {t7_mean:.2f} ms ± {t7_std:.2f} ms per image")
    print(f"Day 10 (Fine-Tuned CNN): {t10_mean:.2f} ms ± {t10_std:.2f} ms per image")
    print(f"Day 16 (Fusion CNN+FFT): {t16_mean:.2f} ms ± {t16_std:.2f} ms per image")
    
    return (t7_mean, t7_std), (t10_mean, t10_std), (t16_mean, t16_std)


# ── Part 2: Precision-Recall & Threshold Investigation ────────────────────── #
def run_pr_investigation(test_df, transform):
    print("\n=======================================================")
    print(" PART 2: PRECISION-RECALL & THRESHOLD ARTIFACT ANALYSIS ")
    print("=======================================================")
    
    day10_model = load_cnn_model(DAY8_CKPT)
    day16_model = load_fusion_model(DAY16_CKPT)
    
    npz = np.load(FREQ_NPZ)
    X_test   = npz["X_test"]
    mean_fft = npz["X_train"].mean(axis=0)
    std_fft  = npz["X_train"].std(axis=0) + 1e-8
    X_test_norm = (X_test - mean_fft) / std_fft

    y_true = np.array([0 if l == "real" else 1 for l in test_df["label"]])
    
    # Get probabilities for Day 10
    probs_day10 = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = FACES_DIR / row["label"] / row["filename"]
            img = Image.open(img_path).convert("RGB")
            t_tensor = transform(img).unsqueeze(0).to(device)
            p = torch.sigmoid(day10_model(t_tensor)).item()
            probs_day10.append(p)
    probs_day10 = np.array(probs_day10)

    # Get probabilities for Day 16
    probs_day16 = []
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            img_path = FACES_DIR / row["label"] / row["filename"]
            img = Image.open(img_path).convert("RGB")
            t_tensor = transform(img).unsqueeze(0).to(device)
            freq_tensor = torch.tensor(X_test_norm[idx], dtype=torch.float32).unsqueeze(0).to(device)
            p = torch.sigmoid(day16_model(t_tensor, freq_tensor)).item()
            probs_day16.append(p)
    probs_day16 = np.array(probs_day16)

    # 1. Default Threshold (0.5) Metrics
    preds_day10_def = (probs_day10 > 0.5).astype(int)
    preds_day16_def = (probs_day16 > 0.5).astype(int)
    
    print("\n--- Default Threshold (0.5) Summary ---")
    print(f"Day 10 (CNN-only): Acc={accuracy_score(y_true, preds_day10_def)*100:.2f}%, Prec={precision_score(y_true, preds_day10_def):.4f}, Rec={recall_score(y_true, preds_day10_def):.4f}, F1={f1_score(y_true, preds_day10_def):.4f}")
    print(f"Day 16 (Fusion):   Acc={accuracy_score(y_true, preds_day16_def)*100:.2f}%, Prec={precision_score(y_true, preds_day16_def):.4f}, Rec={recall_score(y_true, preds_day16_def):.4f}, F1={f1_score(y_true, preds_day16_def):.4f}")

    # 2. PR-AUC / Average Precision
    pr_auc_10 = average_precision_score(y_true, probs_day10)
    pr_auc_16 = average_precision_score(y_true, probs_day16)
    roc_auc_10 = roc_auc_score(y_true, probs_day10)
    roc_auc_16 = roc_auc_score(y_true, probs_day16)

    print("\n--- Threshold-Independent Ranking Metrics ---")
    print(f"Day 10 (CNN-only): ROC-AUC = {roc_auc_10:.4f} | PR-AUC (Avg Precision) = {pr_auc_10:.4f}")
    print(f"Day 16 (Fusion):   ROC-AUC = {roc_auc_16:.4f} | PR-AUC (Avg Precision) = {pr_auc_16:.4f}")

    # 3. Matched Operating Point Analysis
    target_recall = recall_score(y_true, preds_day16_def)  # 0.8667 (130/150)
    print(f"\n--- Matched Operating Point Analysis (Target Recall = {target_recall*100:.2f}%) ---")
    
    p10_prec, p10_rec, p10_thresh = precision_recall_curve(y_true, probs_day10)
    
    # Find index where recall is closest to target_recall
    idx_matched = np.argmin(np.abs(p10_rec - target_recall))
    matched_thresh_10 = p10_thresh[min(idx_matched, len(p10_thresh)-1)]
    
    preds_day10_matched = (probs_day10 >= matched_thresh_10).astype(int)
    prec_10_matched = precision_score(y_true, preds_day10_matched)
    rec_10_matched  = recall_score(y_true, preds_day10_matched)
    acc_10_matched  = accuracy_score(y_true, preds_day10_matched)

    print(f"Day 10 at tuned threshold ({matched_thresh_10:.4f}):")
    print(f"   Accuracy:  {acc_10_matched*100:.2f}%")
    print(f"   Precision: {prec_10_matched:.4f}")
    print(f"   Recall:    {rec_10_matched:.4f}")

    print(f"Day 16 at default threshold (0.5000):")
    print(f"   Accuracy:  {accuracy_score(y_true, preds_day16_def)*100:.2f}%")
    print(f"   Precision: {precision_score(y_true, preds_day16_def):.4f}")
    print(f"   Recall:    {recall_score(y_true, preds_day16_def):.4f}")

    # Plot PR Curves
    plt.figure(figsize=(7, 6))
    prec10, rec10, _ = precision_recall_curve(y_true, probs_day10)
    prec16, rec16, _ = precision_recall_curve(y_true, probs_day16)
    
    plt.plot(rec10, prec10, label=f"Day 10 CNN-only (PR-AUC = {pr_auc_10:.4f})", color="blue", lw=2)
    plt.plot(rec16, prec16, label=f"Day 16 Fusion   (PR-AUC = {pr_auc_16:.4f})", color="orange", lw=2)
    plt.plot([0, 1], [0.5, 0.5], "k--", label="Random Baseline (0.50)")
    
    # Highlight default operating points
    plt.scatter([recall_score(y_true, preds_day10_def)], [precision_score(y_true, preds_day10_def)],
                color="blue", s=80, zorder=5, label="Day 10 default (th=0.50)")
    plt.scatter([recall_score(y_true, preds_day16_def)], [precision_score(y_true, preds_day16_def)],
                color="orange", s=80, zorder=5, label="Day 16 default (th=0.50)")
    # Highlight Day 10 matched operating point
    plt.scatter([rec_10_matched], [prec_10_matched],
                color="navy", marker="X", s=100, zorder=5, label=f"Day 10 tuned (th={matched_thresh_10:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Day 10 (CNN-only) vs Day 16 (Fusion)")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pr_path = RESULTS_DIR / "day16_pr_curve_comparison.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"\nSaved PR Curve Comparison plot to: {pr_path}")

    return {
        "pr_auc_10": pr_auc_10,
        "pr_auc_16": pr_auc_16,
        "roc_auc_10": roc_auc_10,
        "roc_auc_16": roc_auc_16,
        "matched_thresh_10": matched_thresh_10,
        "acc_10_matched": acc_10_matched,
        "prec_10_matched": prec_10_matched,
        "rec_10_matched": rec_10_matched
    }


def main():
    transform = get_val_test_transforms()
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    t7_stats, t10_stats, t16_stats = run_timing_benchmark(test_df, transform)
    pr_results = run_pr_investigation(test_df, transform)
    
    print("\n=======================================================")
    print(" FINAL INVESTIGATION SUMMARY ")
    print("=======================================================")
    print(f"Corrected Single-Image Latency (Batch=1):")
    print(f"  Day 7  (Head-Only CNN):  {t7_stats[0]:.2f} ms ± {t7_stats[1]:.2f} ms")
    print(f"  Day 10 (Fine-Tuned CNN): {t10_stats[0]:.2f} ms ± {t10_stats[1]:.2f} ms")
    print(f"  Day 16 (Fusion CNN+FFT): {t16_stats[0]:.2f} ms ± {t16_stats[1]:.2f} ms")
    print(f"\nPR-AUC (Average Precision):")
    print(f"  Day 10 (CNN-only): {pr_results['pr_auc_10']:.4f}")
    print(f"  Day 16 (Fusion):   {pr_results['pr_auc_16']:.4f} (Diff: {pr_results['pr_auc_16'] - pr_results['pr_auc_10']:+.4f})")
    print(f"\nMatched Recall (86.67%) Precision Comparison:")
    print(f"  Day 10 (tuned th={pr_results['matched_thresh_10']:.4f}): Precision = {pr_results['prec_10_matched']*100:.2f}%, Accuracy = {pr_results['acc_10_matched']*100:.2f}%")
    print(f"  Day 16 (default th=0.5000):   Precision = 75.58%, Accuracy = 79.33%")

if __name__ == "__main__":
    main()
