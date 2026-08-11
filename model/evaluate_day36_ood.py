"""
model/evaluate_day36_ood.py
----------------------------
Day 36: Cross-generator generalization test.

Dataset: "Real vs Fake Faces (StyleGAN3)" — troykueh/real-vs-fake-faces-stylegan3
  - Real faces: FFHQ (same real-photo source as training — isolates generator effect)
  - Fake faces: StyleGAN3 ONLY (completely different generator from StyleGAN2 training)

This tests the core generalization question: does a model trained on StyleGAN2 fakes
transfer to StyleGAN3 fakes? If not, it quantifies exactly how much the detection
capability is generator-specific.

Pipeline: identical to production deployment.
  - Face extraction: Haar Cascade (matching evaluate_haar.py / Render deployment)
  - Classification: day32_finetuned_converged.pth (EfficientNet-B0 CNN-only)
  - No MTCNN — same constraints as the live API

Outputs:
  results/day36_cross_generator_generalization.md
  results/day36_notes.md
"""

import sys
import os
import time
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2
import timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import IMAGENET_MEAN, IMAGENET_STD

CKPT         = ROOT / "model" / "checkpoints" / "day32_finetuned_converged.pth"
RESULTS_DIR  = ROOT / "results"
SG3_DIR      = ROOT / "data" / "day36_sg3_raw"   # extracted dataset root
SAMPLE_DIR   = ROOT / "data" / "day36_sample"    # our 75+75 sample

# Fixed seed for reproducibility
RANDOM_SEED  = 42
N_EACH       = 75   # 75 real + 75 fake = 150 total

# Haar cascade
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(HAAR_XML)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def find_dataset_dirs(root: Path):
    """
    Walk root to find 'real' and 'fake' (or 'Real'/'Fake') subdirectories
    containing images. Returns (real_dir, fake_dir).
    """
    for name in ["real", "Real", "REAL", "0", "Authentic"]:
        candidate = root / name
        if candidate.exists():
            real_dir = candidate
            break
    else:
        raise FileNotFoundError(f"Could not find 'real' subdir under {root}")

    for name in ["fake", "Fake", "FAKE", "1", "Synthetic", "fake_StyleGAN3"]:
        candidate = root / name
        if candidate.exists():
            fake_dir = candidate
            break
    else:
        # Try one level deeper
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        print(f"Subdirs found: {[d.name for d in subdirs]}")
        raise FileNotFoundError(f"Could not find 'fake' subdir under {root}")

    return real_dir, fake_dir


def sample_images(real_dir: Path, fake_dir: Path, n: int, seed: int):
    """Sample n real and n fake image paths, reproducibly."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    real_imgs = sorted([f for f in real_dir.iterdir() if f.is_file() and f.suffix.lower() in exts])
    fake_imgs = sorted([f for f in fake_dir.iterdir() if f.is_file() and f.suffix.lower() in exts])
    print(f"Found {len(real_imgs)} real, {len(fake_imgs)} fake images in dataset")

    rng = random.Random(seed)
    real_sample = rng.sample(real_imgs, min(n, len(real_imgs)))
    fake_sample = rng.sample(fake_imgs, min(n, len(fake_imgs)))
    return real_sample, fake_sample


def extract_face_haar(img_path: Path):
    """
    Extract largest face crop via Haar Cascade.
    Returns PIL Image (RGB) or None on failure (matches production behaviour).
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return None

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_bgr = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def load_model(ckpt_path: Path, device):
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    return model


def run_evaluation(model, image_paths, true_labels, device):
    """
    Run Haar → classify pipeline on each image.
    Detection failures → predicted as 0.5 prob (uncertain), penalised in accuracy.
    """
    probs, preds, labels_out, detect_failures = [], [], [], 0
    times = []

    for img_path, label in zip(image_paths, true_labels):
        face = extract_face_haar(img_path)

        if face is None:
            detect_failures += 1
            # Penalise as wrong prediction (same convention as evaluate_haar.py)
            probs.append(0.5)
            preds.append(1 - label)   # wrong class
            labels_out.append(label)
            continue

        t0 = time.time()
        with torch.no_grad():
            tensor = TRANSFORM(face).unsqueeze(0).to(device)
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()
        times.append(time.time() - t0)

        pred = int(prob > 0.5)
        probs.append(prob)
        preds.append(pred)
        labels_out.append(label)

    avg_ms = (np.mean(times) * 1000) if times else 0.0
    return np.array(labels_out), np.array(probs), np.array(preds), detect_failures, avg_ms


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Locate dataset dirs ──────────────────────────────────────────────── #
    real_dir = SG3_DIR / "Real faces"
    fake_dir = SG3_DIR / "Fake faces"
    print(f"Found dirs — real: {real_dir}, fake: {fake_dir}")

    if not (real_dir.exists() and fake_dir.exists()):
        raise FileNotFoundError(f"Expected dirs not found: {real_dir}, {fake_dir}")

    # ── Sample images ────────────────────────────────────────────────────── #
    real_sample, fake_sample = sample_images(real_dir, fake_dir, N_EACH, RANDOM_SEED)
    actual_n_real = len(real_sample)
    actual_n_fake = len(fake_sample)
    total_n = actual_n_real + actual_n_fake

    # Label convention: 0=REAL, 1=FAKE (same as production model)
    all_paths  = real_sample + fake_sample
    all_labels = [0] * actual_n_real + [1] * actual_n_fake

    print(f"\nOOD sample: {actual_n_real} real + {actual_n_fake} fake = {total_n} total")
    print(f"Random seed: {RANDOM_SEED}\n")

    # ── Load model ───────────────────────────────────────────────────────── #
    model = load_model(CKPT, device)

    # ── Run Haar → classify pipeline ─────────────────────────────────────── #
    print("Running Haar Cascade face detection + classification...")
    y_true, y_prob, y_pred, detect_failures, avg_ms = run_evaluation(
        model, all_paths, all_labels, device
    )

    # ── Metrics ──────────────────────────────────────────────────────────── #
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    # ROC-AUC only valid if both classes present
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    # Accuracy broken down by class (more informative for generalization)
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask])
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask])

    print("\n--- DAY 36 OOD GENERALIZATION: RESULTS ---")
    print(f"Overall Accuracy:     {acc*100:.2f}%")
    print(f"  Real image acc:     {real_acc*100:.2f}%")
    print(f"  Fake image acc:     {fake_acc*100:.2f}%")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"ROC-AUC:              {auc:.4f}")
    print(f"Haar failures:        {detect_failures}/{total_n} ({detect_failures/total_n*100:.1f}%)")
    print(f"Avg inference time:   {avg_ms:.2f} ms/image")
    print(f"Confusion Matrix:\n{cm}")

    # ── In-distribution reference (production numbers from Day 32) ─────── #
    # Production (Haar Cascade, Day 32 CNN-only) on in-distribution test set:
    INDIST_ACC = 0.78
    INDIST_AUC = 0.8387

    # ── Write reports ────────────────────────────────────────────────────── #
    _write_generalization_report(
        acc, prec, rec, f1, auc, cm,
        real_acc, fake_acc, detect_failures, total_n, avg_ms,
        actual_n_real, actual_n_fake, INDIST_ACC, INDIST_AUC
    )
    _write_notes(detect_failures, total_n, actual_n_real, actual_n_fake)
    print(f"\nReports written to {RESULTS_DIR}")


def _write_generalization_report(acc, prec, rec, f1, auc, cm,
                                  real_acc, fake_acc, failures, total_n, avg_ms,
                                  n_real, n_fake, indist_acc, indist_auc):
    delta_acc = (acc - indist_acc) * 100
    delta_auc = auc - indist_auc

    out = RESULTS_DIR / "day36_cross_generator_generalization.md"
    lines = []
    lines.append("# Day 36: Cross-Generator Generalization Test\n")
    lines.append("## Setup\n")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| **Production model** | `day32_finetuned_converged.pth` (EfficientNet-B0, CNN-only) |")
    lines.append("| **Training distribution** | Real: FFHQ; Fake: StyleGAN2 |")
    lines.append("| **OOD real source** | FFHQ (same real-photo source; isolates generator effect) |")
    lines.append("| **OOD fake source** | StyleGAN3 (different generator from StyleGAN2 training) |")
    lines.append("| **Dataset** | `troykueh/real-vs-fake-faces-stylegan3` (Kaggle, CC BY-NC-SA 4.0) |")
    lines.append(f"| **Sample size** | {n_real} real + {n_fake} fake = {n_real+n_fake} total (seed=42) |")
    lines.append("| **Face extraction** | Haar Cascade (identical to Render production pipeline) |")
    lines.append("")
    lines.append("## Results: In-Distribution vs Out-of-Distribution\n")
    lines.append("| Metric | In-Distribution (Day 32, Haar, StyleGAN2) | OOD (Day 36, Haar, StyleGAN3) | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Overall Accuracy** | **78.00%** | **{acc*100:.2f}%** | **{delta_acc:+.2f}pp** |")
    lines.append(f"| Precision | 0.8088 | {prec:.4f} | {prec-0.8088:+.4f} |")
    lines.append(f"| Recall | 0.7333 | {rec:.4f} | {rec-0.7333:+.4f} |")
    lines.append(f"| F1 Score | 0.7692 | {f1:.4f} | {f1-0.7692:+.4f} |")
    lines.append(f"| ROC-AUC | 0.8387 | {auc:.4f} | {delta_auc:+.4f} |")
    lines.append(f"| Haar failures | 4.7% (14/300) | {failures/total_n*100:.1f}% ({failures}/{total_n}) | — |")
    lines.append("")
    lines.append("### Per-Class Accuracy (OOD)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|---|---|")
    lines.append(f"| Real images (FFHQ, OOD source) | {real_acc*100:.2f}% |")
    lines.append(f"| Fake images (StyleGAN3, OOD generator) | {fake_acc*100:.2f}% |")
    lines.append("")
    lines.append("### Confusion Matrix (OOD, 0=Real, 1=Fake)\n")
    lines.append(f"```\n{cm}\n```\n")
    lines.append("*(Rows: True class. Cols: Predicted class.)*\n")
    lines.append("## Interpretation\n")

    # Determine severity for honest reporting
    if delta_acc < -20:
        severity = "**severe** — the model is near-random on StyleGAN3 fakes"
    elif delta_acc < -10:
        severity = "**substantial** — major generalization gap"
    elif delta_acc < -5:
        severity = "**moderate** — meaningful generalization gap"
    elif delta_acc < 0:
        severity = "**mild** — small generalization gap"
    else:
        severity = "**none** — model generalizes to StyleGAN3 (unexpected positive result)"

    lines.append(f"The accuracy drop from in-distribution to OOD is {severity}.")
    lines.append(f"\nDelta: **{delta_acc:+.2f} percentage points** (ROC-AUC: {delta_auc:+.4f}).")
    lines.append("\nPer-class breakdown:")
    lines.append(f"- Real image accuracy: **{real_acc*100:.2f}%** — measures whether the model still correctly identifies real photos (Unsplash/FFHQ real-world photos vs FFHQ training real photos).")
    lines.append(f"- Fake image accuracy: **{fake_acc*100:.2f}%** — the critical number: this directly measures whether StyleGAN2-trained detection transfers to StyleGAN3.")
    lines.append("")
    lines.append("### What This Means\n")
    lines.append("The model was trained exclusively on StyleGAN2-generated fakes paired with FFHQ real faces.")
    lines.append("StyleGAN2 and StyleGAN3 differ significantly in their generation mechanism:")
    lines.append("- **StyleGAN2** uses transposed convolutions → characteristic grid-pattern spectral artefacts at specific frequencies")
    lines.append("- **StyleGAN3** uses alias-free synthesis with continuous coordinates → different (or suppressed) frequency artefacts")
    lines.append("")
    lines.append("If fake_acc drops sharply, the model has learned StyleGAN2-specific spectral signatures rather than general GAN-detection features. This is expected and scientifically important.")
    lines.append("")
    lines.append("**Scope conclusion:** This result explicitly quantifies the model's generalization boundary.")
    lines.append("The detection capability is scoped to faces from the StyleGAN2 training distribution.")
    lines.append("Claims of 'AI-generated face detection' should be qualified as 'StyleGAN2-generated face detection' in research contexts.")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generalization report: {out}")


def _write_notes(failures, total_n, n_real, n_fake):
    out = RESULTS_DIR / "day36_notes.md"
    lines = []
    lines.append("# Day 36 Notes: Cross-Generator Generalization Test\n")
    lines.append("## Data Sources\n")
    lines.append("### Primary OOD Dataset (Used)\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| **Dataset name** | Real vs Fake Faces (StyleGAN3) |")
    lines.append("| **Kaggle URL** | https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3 |")
    lines.append("| **License** | CC BY-NC-SA 4.0 |")
    lines.append("| **Real source** | FFHQ (same as training real set) |")
    lines.append("| **Fake source** | StyleGAN3 ONLY — clean, single-generator labelling |")
    lines.append("| **Why chosen** | Isolates generator effect: real-photo source is held constant (FFHQ), so any accuracy drop is attributable entirely to the generator change (StyleGAN2→StyleGAN3), not to a different real-photo domain |")
    lines.append(f"| **Sample size** | {n_real} real + {n_fake} fake (seed=42 fixed) |")
    lines.append("")
    lines.append("### Rejected Dataset: `chuneeb/deepfake-detection-dataset-2026`\n")
    lines.append("This dataset was initially downloaded but found to be **unusable** for evaluation:")
    lines.append("- The dataset labels 3,767 images from `randomuser.me/api/portraits/` as FAKE (StyleGAN3)")
    lines.append("- `randomuser.me` serves **real stock photographs** of real people as API avatars, not AI-generated images")
    lines.append("- Using this dataset would test nothing meaningful: both classes contain real human photos")
    lines.append("- This is a data quality error in the Kaggle dataset, not a flaw in our pipeline")
    lines.append("- **Decision:** discarded; switched to `troykueh/real-vs-fake-faces-stylegan3` which has verified StyleGAN3 outputs\n")
    lines.append("## Methodology\n")
    lines.append("- **Face extraction:** Haar Cascade (identical to Render production pipeline — `evaluate_haar.py` convention)")
    lines.append("- **Model:** `day32_finetuned_converged.pth` — the deployed production model")
    lines.append(f"- **Haar detection failures:** {failures}/{total_n} ({failures/total_n*100:.1f}%) — penalised as wrong predictions (same methodology as Day 31/32 Haar delta measurement)")
    lines.append("- **Label convention:** 0=REAL, 1=FAKE (consistent with all prior evaluation)")
    lines.append("- **Random seed:** 42 (fixed for reproducibility)")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Notes: {out}")


if __name__ == "__main__":
    main()
