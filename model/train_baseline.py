"""
model/train_baseline.py
-----------------------
Day 6: EfficientNet-B0 head-only baseline training.

Architecture:
    EfficientNet-B0 pretrained on ImageNet (via timm), backbone frozen.
    Final classification head replaced: Linear(1280 -> 1, bias=True).

Output convention (matches dataset.py):
    Raw logit > 0  =>  predicted FAKE  (label=1)
    Raw logit <= 0 =>  predicted REAL  (label=0)

Loss function: BCEWithLogitsLoss
    Chosen over CrossEntropyLoss because:
    - Output is a single logit (not 2-class softmax), which is numerically
      more stable and requires one fewer output neuron for a binary task.
    - Gradient flows directly through sigmoid + BCE in one fused operation.

Optimizer: Adam, lr=1e-3
    Standard choice for head-only training where only the new random weights
    need to be learned quickly.

Checkpoint strategy:
    Save ONLY the best checkpoint by val_accuracy (ties broken by lower val_loss).
    Rationale: val_accuracy is the metric we care about for balanced binary tasks,
    and the dataset is perfectly balanced (50/50) so accuracy == balanced accuracy.

Run from project root:
    python model/train_baseline.py [--epochs N] [--batch-size B] [--lr LR]
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import timm

# ── Path setup ────────────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_train_transforms, get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
SPLITS_DIR   = ROOT / "data" / "splits"
FACES_DIR    = ROOT / "data" / "faces_extracted"
CKPT_DIR     = ROOT / "model" / "checkpoints"
LOG_DIR      = ROOT / "results" / "logs"
BEST_CKPT    = CKPT_DIR / "day6_head_only_best.pth"


# ── Model ─────────────────────────────────────────────────────────────────── #
def build_model() -> nn.Module:
    """
    Load EfficientNet-B0 pretrained on ImageNet, freeze backbone, replace head.
    """
    model = timm.create_model("efficientnet_b0", pretrained=True)

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head (1280 -> 1 binary logit)
    in_features = model.classifier.in_features   # 1280 for EfficientNet-B0
    model.classifier = nn.Linear(in_features, 1)  # Only this layer will train

    # Confirm frozen/unfrozen counts
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B0 (timm)")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}  (head only)")
    print(f"  Frozen params:    {total - trainable:,}  (backbone)")

    return model


# ── DataLoaders ───────────────────────────────────────────────────────────── #
def make_loaders(batch_size: int):
    def _loader(split, transform, shuffle):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        ds = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _loader("train", get_train_transforms(), shuffle=True),
        _loader("val",   get_val_test_transforms(), shuffle=False),
    )


# ── Epoch helpers ─────────────────────────────────────────────────────────── #
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    n = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)   # (B,) -> (B,1) for BCEWithLogitsLoss

            logits = model(images)                             # (B,1)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (logits.detach() > 0).long()               # threshold at 0
            correct += (preds.squeeze(1) == labels.squeeze(1).long()).sum().item()
            n += images.size(0)

    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser(description="Day 6: EfficientNet-B0 head-only training")
    parser.add_argument("--epochs",     type=int,   default=10,   help="Number of epochs")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate for head")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model     = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    train_loader, val_loader = make_loaders(args.batch_size)
    print(f"\nDataset sizes: train={len(train_loader.dataset)} | val={len(val_loader.dataset)}\n")

    log_path = LOG_DIR / "day6_training_log.csv"
    log_rows  = []

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    header = f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        elapsed = time.time() - t0
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | {val_loss:>8.4f} | {val_acc:>6.2%} | {elapsed:>5.1f}s")

        # Save best checkpoint (by val_accuracy, ties by val_loss)
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":    val_acc,
                "val_loss":   val_loss,
                "config":     vars(args),
            }, BEST_CKPT)
            print(f"         -> Best checkpoint saved (val_acc={val_acc:.4f})")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "elapsed_s":  round(elapsed,    1),
        })

    # Write CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining log saved to {log_path}")
    print(f"Best checkpoint:  val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
    print(f"Checkpoint saved to: {BEST_CKPT}")


if __name__ == "__main__":
    main()
