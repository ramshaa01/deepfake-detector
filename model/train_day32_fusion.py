"""
model/train_day32_fusion.py
----------------------------
Day 32: Train the fusion head on top of the newly-converged CNN backbone.

Mirrors train_fusion.py exactly, but uses:
  - CNN_CKPT  = day32_finetuned_converged.pth   (new converged backbone)
  - FUSION_CKPT = day32_fusion_converged.pth      (new output)
  - Log: results/logs/day32_fusion_training_log.csv

Run from project root:
    python model/train_day32_fusion.py [--epochs N]
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

SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
FREQ_NPZ    = ROOT / "data" / "frequency_features.npz"
CKPT_DIR    = ROOT / "model" / "checkpoints"
LOG_DIR     = ROOT / "results" / "logs"
CNN_CKPT    = CKPT_DIR / "day32_finetuned_converged.pth"   # ← new converged backbone
FUSION_CKPT = CKPT_DIR / "day32_fusion_converged.pth"       # ← new output

FREQ_LR   = 1e-3
ES_PATIENCE = 5


class FusionDataset(Dataset):
    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, df, faces_dir, freq_features, transform=None):
        self.df = df.reset_index(drop=True)
        self.faces_dir = Path(faces_dir)
        self.transform = transform
        freq = freq_features.astype(np.float32)
        mean = freq.mean(axis=0); std = freq.std(axis=0) + 1e-8
        self.freq_norm = (freq - mean) / std

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.faces_dir / row["label"] / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        freq  = torch.tensor(self.freq_norm[idx], dtype=torch.float32)
        label = torch.tensor(self.LABEL_MAP[row["label"]], dtype=torch.long)
        return image, freq, label


class FreqBranch(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, cnn_backbone):
        super().__init__()
        self.cnn = cnn_backbone
        self.freq_branch = FreqBranch()
        self.head = nn.Linear(1280 + 16, 1)

    def forward(self, images, freq_feat):
        cnn_feat = self.cnn(images)
        freq_out = self.freq_branch(freq_feat)
        return self.head(torch.cat([cnn_feat, freq_out], dim=1))


def build_fusion_model(device):
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)

    print(f"Loading CNN checkpoint: {CNN_CKPT}")
    ckpt = torch.load(CNN_CKPT, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    backbone.classifier = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False

    model = FusionModel(backbone).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}  (freq branch + fusion head only)")
    return model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, freq, labels in loader:
            images = images.to(device); freq = freq.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            logits = model(images, freq)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = (logits.detach() > 0).long()
            correct += (preds.squeeze(1) == labels.squeeze(1).long()).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Day 32: Train fusion head on converged CNN")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading frequency features from: {FREQ_NPZ}")
    npz = np.load(FREQ_NPZ)
    X_train = npz["X_train"]; X_val = npz["X_val"]
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
    train_ds = FusionDataset(train_df, FACES_DIR, X_train, transform=get_train_transforms())
    val_ds   = FusionDataset(val_df,   FACES_DIR, X_val,   transform=get_val_test_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: train={len(train_ds)} | val={len(val_ds)}\n")

    model     = build_fusion_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=FREQ_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_acc = 0.0; best_val_loss = float("inf")
    best_epoch = 0; es_counter = 0; log_rows = []

    header = (f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}")
    print(header); print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, False)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%} | {lr:>8.2e} | {elapsed:>5.1f}s")

        scheduler.step(val_loss)

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc; best_val_loss = val_loss; best_epoch = epoch; es_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "val_loss": val_loss}, FUSION_CKPT)
            print(f"         -> Best checkpoint saved (val_acc={val_acc:.4f})")
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss; es_counter = 0
            else:
                es_counter += 1

        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                          "train_acc": round(train_acc, 6), "val_loss": round(val_loss, 6),
                          "val_acc": round(val_acc, 6), "lr": lr,
                          "elapsed_s": round(elapsed, 1), "es_counter": es_counter})

        # Write incrementally
        log_path = LOG_DIR / "day32_fusion_training_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader(); writer.writerows(log_rows)

        if es_counter >= ES_PATIENCE:
            print(f"\n[Early Stopping] No improvement for {ES_PATIENCE} epochs. Stopping at epoch {epoch}.")
            break

    print(f"\nTraining complete. Best: epoch={best_epoch}, val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {FUSION_CKPT}")


if __name__ == "__main__":
    main()
