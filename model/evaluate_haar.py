"""
model/evaluate_haar.py
------------------------
Day 31: Evaluate the fusion model on the test set using the production 
Haar Cascade face detector instead of MTCNN.

Quantifies the real-world deployment accuracy drop caused by using a 
lighter detector.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import timm
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import get_val_test_transforms
from model.train_fusion import FusionModel, FUSION_CKPT, SPLITS_DIR, FREQ_NPZ
from model.frequency_features import extract_radial_profile

# Import production logic directly
from inference.main import haar_extract_face

def build_model_from_ckpt(ckpt_path, device):
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False

    model = FusionModel(backbone).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    model = build_model_from_ckpt(FUSION_CKPT, device)
    transform = get_val_test_transforms()

    print("Loading test data...")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    # Load frequency stats
    npz = np.load(FREQ_NPZ)
    X_test = npz["X_test"]
    FREQ_MEAN = X_test.mean(axis=0)
    FREQ_STD = X_test.std(axis=0) + 1e-8

    all_labels = []
    all_preds = []
    all_probs = []
    failures = 0

    model.eval()
    
    tmp_path = ROOT / "model" / "tmp_face.jpg"
    
    print(f"Evaluating {len(test_df)} samples using Haar Cascade...")
    
    for idx, row in test_df.iterrows():
        source_path = ROOT / row["source_path"]
        label_str = row["label"].lower()
        true_label = 0 if label_str == "real" else 1
        
        with open(source_path, "rb") as f:
            img_bytes = f.read()
            
        face_pil = haar_extract_face(img_bytes)
        
        if face_pil is None:
            failures += 1
            # In production this is a 400 error, so it's effectively a missed prediction.
            # To penalize accuracy fairly, we count it as a wrong prediction.
            # E.g. if true_label=0 (real), we predict 1. If true_label=1 (fake), we predict 0.
            # This ensures detection failures drop the overall accuracy metric.
            pred_label = 1 if true_label == 0 else 0
            prob = 1.0 if pred_label == 1 else 0.0
        else:
            img_tensor = transform(face_pil).unsqueeze(0).to(device)
            face_pil.save(str(tmp_path), format="JPEG")
            freq_raw = extract_radial_profile(str(tmp_path))
            freq_norm = (freq_raw.astype(np.float32) - FREQ_MEAN) / FREQ_STD
            freq_tensor = torch.tensor(freq_norm, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logit = model(img_tensor, freq_tensor)
                prob = torch.sigmoid(logit).item()
                pred_label = 1 if prob > 0.5 else 0
                
        all_labels.append(true_label)
        all_preds.append(pred_label)
        all_probs.append(prob)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)}")

    if tmp_path.exists():
        tmp_path.unlink()

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    
    print("\n--- HAAR CASCADE PRODUCTION PIPELINE RESULTS ---")
    print(f"Total Test Images: {len(test_df)}")
    print(f"Detection Failures:{failures} ({failures/len(test_df)*100:.1f}%)")
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")

    out_path = ROOT / "results" / "day31_detector_delta.md"
    
    with open(out_path, "w") as f:
        f.write("# Day 31: Production Detector Delta (MTCNN vs Haar Cascade)\n\n")
        f.write("This evaluates the real-world deployment accuracy drop caused by using the lightweight Haar Cascade detector instead of MTCNN (due to 512MB RAM constraints on the Render free tier).\n\n")
        f.write(f"**Test Set:** {len(test_df)} images (balanced).\n")
        f.write(f"**Haar Cascade Detection Failures (400 Bad Request):** {failures} ({failures/len(test_df)*100:.1f}%)\n")
        f.write("*Note: Detection failures are treated as incorrect predictions for the Haar metrics to reflect true system accuracy.* \n\n")
        
        f.write("| Metric | Day 16 (MTCNN - Official) | Day 31 (Haar Cascade - Production) | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Accuracy | 79.33% | {acc*100:.2f}% | {acc*100 - 79.33:+.2f}% |\n")
        f.write(f"| Precision | 0.7558 | {prec:.4f} | {prec - 0.7558:+.4f} |\n")
        f.write(f"| Recall | 0.8667 | {rec:.4f} | {rec - 0.8667:+.4f} |\n")
        f.write(f"| F1 Score | 0.8075 | {f1:.4f} | {f1 - 0.8075:+.4f} |\n")
        f.write(f"| ROC-AUC | 0.8544 | {auc:.4f} | {auc - 0.8544:+.4f} |\n")

    print(f"\nDelta metrics saved to {out_path}")

if __name__ == "__main__":
    main()
