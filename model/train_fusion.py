"""
model/train_fusion.py
---------------------
Day 16: CNN + Frequency Feature Fusion Model

Architecture:
  - CNN Branch: frozen fine-tuned EfficientNet-B0 (day8_finetuned_best.pth)
    used as a fixed feature extractor — outputs a 1280-dim penultimate feature.
  - Frequency Branch: small MLP (16 -> 32 -> 16) that processes the FFT radial
    energy feature computed in Day 15.
  - Fusion Head: concatenates CNN (1280-dim) + Freq (16-dim) -> Linear -> 1 output.

Only the frequency branch + fusion head are trained; the CNN backbone is FROZEN.
This keeps the comparison fair and avoids expensive retraining.

Frequency features are loaded from the pre-computed .npz (data/frequency_features.npz)
generated in Day 15 — no FFT recomputation needed at training time.

Run from project root:
    python model/train_fusion.py [--epochs N] [--batch-size B]
"""

import sys
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import get_train_transforms, get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
FREQ_NPZ    = ROOT / "data" / "frequency_features.npz"  # pre-computed Day 15
CKPT_DIR    = ROOT / "model" / "checkpoints"
LOG_DIR     = ROOT / "results" / "logs"
CNN_CKPT    = CKPT_DIR / "day8_finetuned_best.pth"
FUSION_CKPT = CKPT_DIR / "day16_fusion_best.pth"

FREQ_LR = 1e-3


# ── Fusion Dataset ─────────────────────────────────────────────────────────── #
class FusionDataset(Dataset):
    """
    Returns (image_tensor, freq_feature, label) for each sample.
    Frequency features are loaded from the pre-computed .npz (Day 15)
    aligned by row index with the split CSVs — instant, no recomputation.
    """
    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, df, faces_dir, freq_features, transform=None):
        self.df = df.reset_index(drop=True)
        self.faces_dir = Path(faces_dir)
        self.transform = transform

        # Normalize frequency features (per-bin, zero mean / unit std)
        freq = freq_features.astype(np.float32)
        mean = freq.mean(axis=0)
        std  = freq.std(axis=0) + 1e-8
        self.freq_norm = (freq - mean) / std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.faces_dir / row["label"] / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        freq  = torch.tensor(self.freq_norm[idx], dtype=torch.float32)
        label = torch.tensor(self.LABEL_MAP[row["label"]], dtype=torch.long)
        return image, freq, label


# ── Model Architecture ─────────────────────────────────────────────────────── #
class FreqBranch(nn.Module):
    """Tiny MLP for the 16-bin FFT feature: 16 -> 32 -> 16"""
    def __init__(self, in_dim=16, hidden=32, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    """
    Frozen CNN feature extractor + small Frequency MLP + fusion head.

    CNN features:   1280-dim (EfficientNet-B0 after global pool, classifier=Identity)
    Freq features:    16-dim (output of FreqBranch MLP)
    Concatenated:   1296-dim -> Linear(1)
    """
    def __init__(self, cnn_backbone):
        super().__init__()
        self.cnn = cnn_backbone
        self.freq_branch = FreqBranch()
        self.head = nn.Linear(1280 + 16, 1)

    def forward(self, images, freq_feat):
        cnn_feat = self.cnn(images)              # (B, 1280)
        freq_out = self.freq_branch(freq_feat)   # (B, 16)
        combined = torch.cat([cnn_feat, freq_out], dim=1)  # (B, 1296)
        return self.head(combined)


def build_fusion_model(device):
    """
    Load day8_finetuned_best.pth, replace classifier with Identity (so forward
    returns the 1280-dim penultimate feature), FREEZE all CNN weights, and
    attach the frequency branch + fusion head.
    """
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)   # needed to match ckpt keys

    print(f"Loading CNN checkpoint: {CNN_CKPT}")
    ckpt = torch.load(CNN_CKPT, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    # Strip the single-neuron head so forward() returns the penultimate features
    backbone.classifier = nn.Identity()

    # FREEZE all CNN backbone weights
    for param in backbone.parameters():
        param.requires_grad = False

    model = FusionModel(backbone).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}  (freq branch + fusion head only)")

    return model


# ── Epoch helpers ─────────────────────────────────────────────────────────── #
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, freq, labels in loader:
            images = images.to(device)
            freq   = freq.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images, freq)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds   = (logits.detach() > 0).long()
            correct += (preds.squeeze(1) == labels.squeeze(1).long()).sum().item()
            n       += images.size(0)

    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser(description="Day 16: CNN + Frequency Feature Fusion")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: max_epochs={args.epochs}, batch_size={args.batch_size}, freq_lr={FREQ_LR}\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not CNN_CKPT.exists():
        print(f"ERROR: CNN checkpoint not found at {CNN_CKPT}"); sys.exit(1)
    if not FREQ_NPZ.exists():
        print(f"ERROR: Frequency features .npz not found at {FREQ_NPZ}"); sys.exit(1)

    # ── Load pre-computed frequency features ─────────────────────────────── #
    print(f"Loading pre-computed frequency features from: {FREQ_NPZ}")
    npz = np.load(FREQ_NPZ)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val,   y_val   = npz["X_val"],   npz["y_val"]
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    # ── Datasets ─────────────────────────────────────────────────────────── #
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")

    train_ds = FusionDataset(train_df, FACES_DIR, X_train, transform=get_train_transforms())
    val_ds   = FusionDataset(val_df,   FACES_DIR, X_val,   transform=get_val_test_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: train={len(train_ds)} | val={len(val_ds)}\n")

    # ── Model, loss, optimiser ──────────────────────────────────────────── #
    model     = build_fusion_model(device)
    criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=FREQ_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # ── Training loop ───────────────────────────────────────────────────── #
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    es_counter    = 0
    ES_PATIENCE   = 5
    log_rows      = []

    header = (f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%} | {lr:>8.2e} | {elapsed:>5.1f}s")

        scheduler.step(val_loss)

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "val_loss":         val_loss,
            }, FUSION_CKPT)
            print(f"         -> Best checkpoint saved (val_acc={val_acc:.4f})")

        if val_loss < best_val_loss or is_best:
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= ES_PATIENCE:
            print(f"\n[Early Stopping] No improvement for {ES_PATIENCE} epochs. Stopping.")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         lr,
            "elapsed_s":  round(elapsed, 1),
        })

        if es_counter >= ES_PATIENCE:
            break

    log_path = LOG_DIR / "day16_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining log: {log_path}")
    print(f"Best checkpoint: epoch={best_epoch}, val_acc={best_val_acc:.4f}")
    print(f"Checkpoint path: {FUSION_CKPT}")


if __name__ == "__main__":
    main()
