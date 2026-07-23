"""
model/gradcam_compare.py
------------------------
Day 14: Grad-CAM Variant Comparison

Compares Grad-CAM, Grad-CAM++, and Eigen-CAM on a selection of images:
- Correctly classified reals/fakes
- Specific glasses-related False Positives (to cross-validate the spurious bias)

Run from project root:
    python model/gradcam_compare.py
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
import pandas as pd

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Path setup ────────────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
FACES_DIR   = ROOT / "data" / "faces_extracted"
CKPT_PATH   = ROOT / "model" / "checkpoints" / "day8_finetuned_best.pth"
RESULTS_DIR = ROOT / "results"
SUMMARY_IMG = RESULTS_DIR / "day14_cam_variant_comparison.png"

# Hardcoded lists of interesting images
# FPs related to glasses
GLASSES_FPS = [
    ("30933.jpg", "real"),
    ("53220.jpg", "real"),
    ("41035.jpg", "real"),
    ("17734.jpg", "real")
]

# We will dynamically find 4 correctly classified images (2 real, 2 fake)
# to compare as well.

# ── Model Loading ─────────────────────────────────────────────────────────── #
def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def process_cam_variants(filename, label_dir, model, target_layers, device, transform, cams):
    img_path = FACES_DIR / label_dir / filename
    if not img_path.exists():
        return None
        
    img_pil = Image.open(img_path).convert('RGB')
    orig_img_np = np.array(img_pil.resize((224, 224))) / 255.0
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()
        pred = "fake" if prob > 0.5 else "real"
        
    targets = [ClassifierOutputTarget(0)]
    
    overlays = []
    for name, cam in cams.items():
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
        overlay = show_cam_on_image(orig_img_np, grayscale_cam, use_rgb=True)
        overlays.append((name, overlay))
        
    return orig_img_np, overlays, prob, pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CKPT_PATH, device)
    target_layers = [model.conv_head]
    transform = get_val_test_transforms()
    
    # Initialize CAM variants
    cams = {
        "Grad-CAM": GradCAM(model=model, target_layers=target_layers),
        "Grad-CAM++": GradCAMPlusPlus(model=model, target_layers=target_layers),
        "Eigen-CAM": EigenCAM(model=model, target_layers=target_layers)
    }
    
    print("\n--- Identifying Correct Samples ---")
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    
    selected_correct = []
    for label in ["real", "fake"]:
        subset = test_df[test_df["label"] == label].sample(frac=1, random_state=42)
        count = 0
        for _, row in subset.iterrows():
            if count >= 2:
                break
            img_tensor = transform(Image.open(FACES_DIR / label / row["filename"]).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(img_tensor)).item()
                pred = "fake" if prob > 0.5 else "real"
            if pred == label:
                selected_correct.append((row["filename"], label))
                count += 1
                
    # Combine the lists
    images_to_test = GLASSES_FPS + selected_correct
    
    fig, axes = plt.subplots(len(images_to_test), 4, figsize=(16, 4 * len(images_to_test)))
    
    for row_idx, (filename, true_label) in enumerate(images_to_test):
        orig, overlays, prob, pred_label = process_cam_variants(filename, true_label, model, target_layers, device, transform, cams)
        
        # Column 0: Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig)
        color = "green" if true_label == pred_label else "red"
        ax_orig.set_title(f"True: {true_label.upper()}\nPred: {pred_label.upper()}\nProb(Fake): {prob:.2%}", color=color)
        ax_orig.axis("off")
        
        # Columns 1-3: CAM Variants
        for col_idx, (name, overlay) in enumerate(overlays):
            ax_cam = axes[row_idx, col_idx + 1]
            ax_cam.imshow(overlay)
            ax_cam.set_title(name)
            ax_cam.axis("off")
            
    fig.suptitle("Grad-CAM Variant Comparison (Grad-CAM vs Grad-CAM++ vs Eigen-CAM)", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(SUMMARY_IMG, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison grid to: {SUMMARY_IMG}")

if __name__ == "__main__":
    main()
