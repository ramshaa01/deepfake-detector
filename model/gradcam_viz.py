"""
model/gradcam_viz.py
--------------------
Day 13: Grad-CAM Integration

Generates Grad-CAM heatmaps for specified images using the fine-tuned
EfficientNet-B0 model. We use the last convolutional block (model.conv_head)
as the target layer since it contains the highest-level spatial features
before the global average pooling.

Since this is a binary classifier with a single output logit (1 = fake),
we use ClassifierOutputTarget(0) to compute gradients with respect to this logit.
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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Path setup ────────────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
FACES_DIR   = ROOT / "data" / "faces_extracted"
CKPT_PATH   = ROOT / "model" / "checkpoints" / "day8_finetuned_best.pth"
CORRECT_DIR = ROOT / "results" / "gradcam_correct_samples"
ERROR_DIR   = ROOT / "results" / "gradcam_error_samples"
SUMMARY_IMG = ROOT / "results" / "day13_gradcam_summary.png"

# Hardcoded lists based on Day 12 error analysis & general correct samples
# Format: (filename, label_dir)
HARDEST_ERRORS = [
    ("30933.jpg", "real"),  # FP (glasses/older)
    ("53220.jpg", "real"),  # FP
    ("41035.jpg", "real"),  # FP
    ("17734.jpg", "real"),  # FP
    ("58LQXHRX1N.jpg", "fake"), # FN (harsh lighting/shadows)
    ("UM7Z524GZT.jpg", "fake"), # FN
    ("P76Z3AI1S6.jpg", "fake"), # FN
    ("ZO99XXQ7U0.jpg", "fake")  # FN
]

CORRECT_SAMPLES = [
    # Just picking some generic names that might exist, 
    # but to be safe we can just find random correctly classified ones dynamically.
    # Actually, let's just pick some random images and verify if they are correct.
    # To keep this script simple and robust, we'll scan the test split.
]

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


def generate_cam(model, target_layers, img_tensor, orig_img_np):
    """
    Generates Grad-CAM overlay.
    img_tensor: [1, C, H, W] tensor ready for the model
    orig_img_np: [H, W, 3] float numpy array [0, 1] for overlay
    """
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Target 0 because we have a single output node (binary)
    targets = [ClassifierOutputTarget(0)]
    
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(orig_img_np, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def process_image(filename, label_dir, model, target_layers, device, transform):
    img_path = FACES_DIR / label_dir / filename
    if not img_path.exists():
        print(f"Warning: {img_path} not found.")
        return None, None, None, None

    # Load image for overlay (RGB, [0, 1])
    img_pil = Image.open(img_path).convert('RGB')
    orig_img_np = np.array(img_pil.resize((224, 224))) / 255.0
    
    # Transform for model
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()
        pred = "fake" if prob > 0.5 else "real"
        
    # Generate CAM
    overlay, _ = generate_cam(model, target_layers, img_tensor, orig_img_np)
    
    return orig_img_np, overlay, prob, pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CKPT_PATH, device)
    
    # EfficientNet-B0 last conv layer in timm
    target_layers = [model.conv_head]
    transform = get_val_test_transforms()
    
    CORRECT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Processing Hardest Errors ---")
    error_overlays = []
    for filename, label in HARDEST_ERRORS:
        orig, overlay, prob, pred = process_image(filename, label, model, target_layers, device, transform)
        if overlay is not None:
            out_path = ERROR_DIR / f"{label}_{pred}_{filename}"
            Image.fromarray(overlay).save(out_path)
            error_overlays.append((orig, overlay, label, pred, prob))
            print(f"Saved: {out_path} (True: {label}, Pred: {pred}, Prob_fake: {prob:.2f})")
            
    print("\n--- Processing Correct Samples ---")
    import pandas as pd
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    
    correct_overlays = []
    # Find 4 correct Reals and 4 correct Fakes dynamically
    for label in ["real", "fake"]:
        subset = test_df[test_df["label"] == label].sample(frac=1, random_state=42)
        count = 0
        for _, row in subset.iterrows():
            if count >= 4:
                break
            orig, overlay, prob, pred = process_image(row["filename"], label, model, target_layers, device, transform)
            if overlay is not None and pred == label:
                out_path = CORRECT_DIR / f"{label}_{pred}_{row['filename']}"
                Image.fromarray(overlay).save(out_path)
                correct_overlays.append((orig, overlay, label, pred, prob))
                print(f"Saved: {out_path} (True: {label}, Pred: {pred}, Prob_fake: {prob:.2f})")
                count += 1

    print("\n--- Generating Summary Grid ---")
    # Pick 2 correct real, 2 correct fake, 2 FP, 2 FN
    summary_items = []
    # Correct Real
    summary_items.extend([item for item in correct_overlays if item[2] == "real"][:2])
    # Correct Fake
    summary_items.extend([item for item in correct_overlays if item[2] == "fake"][:2])
    # FP
    summary_items.extend([item for item in error_overlays if item[2] == "real" and item[3] == "fake"][:2])
    # FN
    summary_items.extend([item for item in error_overlays if item[2] == "fake" and item[3] == "real"][:2])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (orig, overlay, true_label, pred_label, prob) in enumerate(summary_items):
        ax = axes[i]
        
        # We'll plot the overlay
        ax.imshow(overlay)
        
        color = "green" if true_label == pred_label else "red"
        title = f"True: {true_label.upper()}\nPred: {pred_label.upper()}\nProb(Fake): {prob:.2%}"
        ax.set_title(title, color=color, fontweight="bold")
        ax.axis("off")
        
    fig.suptitle("Grad-CAM Interpretability: Correct vs Error Samples", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SUMMARY_IMG, dpi=150, bbox_inches="tight")
    print(f"Saved summary grid to: {SUMMARY_IMG}")


if __name__ == "__main__":
    main()
