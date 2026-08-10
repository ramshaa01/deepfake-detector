"""
model/train_xception.py
-----------------------
Day 34b: XceptionNet Full Fine-Tuning.

Loads the day34a head-only checkpoint, unfreezes the backbone, and trains
end-to-end with differential learning rates until convergence.

Outputs:
    model/checkpoints/day34_xception_best.pth
    results/logs/day34b_xception_finetune_log.csv
"""

import os
import sys
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import timm
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import DeepfakeFaceDataset, IMAGENET_MEAN, IMAGENET_STD

# ── Constants ─────────────────────────────────────────────────────────────── #
SPLITS_DIR = ROOT / "data" / "splits"
FACES_DIR  = ROOT / "data" / "faces_extracted"
CKPT_DIR   = ROOT / "model" / "checkpoints"
LOG_DIR    = ROOT / "results" / "logs"
START_CKPT = CKPT_DIR / "day34a_xception_headonly_best.pth"
BEST_CKPT  = CKPT_DIR / "day34_xception_best.pth"
LOG_PATH   = LOG_DIR  / "day34b_xception_finetune_log.csv"

XCEPTION_SIZE = 299
BACKBONE_LR   = 1e-5
HEAD_LR       = 1e-4
MAX_EPOCHS    = 30
ES_PATIENCE   = 4
BATCH_SIZE    = 32

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((XCEPTION_SIZE, XCEPTION_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((XCEPTION_SIZE, XCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def make_loaders():
    # Use subset for test environment simulation (fast run)
    def _loader(split, transform, shuffle):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv").head(16)
        ds = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
    return _loader("train", get_train_transforms(), True), _loader("val", get_val_transforms(), False)

def build_xception(checkpoint_path):
    model = timm.create_model("legacy_xception", pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    print(f"Loading weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    return model

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = (logits.detach() > 0).long()
            correct += (preds.squeeze(1) == labels.squeeze(1).long()).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n

def main():
    print("Starting fine-tuning script...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = build_xception(START_CKPT).to(device)
    train_loader, val_loader = make_loaders()
    criterion = nn.BCEWithLogitsLoss()

    head_ids = {id(p) for p in model.fc.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params     = list(model.fc.parameters())

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ])

    best_val_acc, best_val_loss = 0.0, float("inf")
    es_counter = 0
    log_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        elapsed = time.time() - t0

        print(f"Ep {epoch:>2} | TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.2%} | VaLoss: {val_loss:.4f} | VaAcc: {val_acc:.2%} | Time: {elapsed:.1f}s")
        
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        
        # Checkpoint EVERY epoch
        ckpt_data = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc, "val_loss": val_loss}
        torch.save(ckpt_data, CKPT_DIR / f"day34b_epoch_{epoch:02d}.pth")
        
        if is_best:
            best_val_acc, best_val_loss = val_acc, val_loss
            torch.save(ckpt_data, BEST_CKPT)
            es_counter = 0
            print("  -> New best checkpoint saved!")
        else:
            es_counter += 1

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if es_counter >= ES_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

if __name__ == "__main__":
    main()
