"""
model/train_finetune.py
-----------------------
Day 8: Full EfficientNet-B0 Fine-Tuning (end-to-end).

Starting point: model/checkpoints/day6_head_only_best.pth
Strategy:
    - Load the pre-trained head-only checkpoint (avoids cold-starting the head)
    - Unfreeze all backbone layers
    - Use DIFFERENTIAL learning rates:
        backbone: lr=1e-5  (small — preserve ImageNet features, avoid catastrophic forgetting)
        head:     lr=1e-4  (10x larger — head can move faster, already partially trained)
    - Scheduler: ReduceLROnPlateau (on val_loss, factor=0.5, patience=2)
        Chosen over cosine annealing because:
        * Small dataset (1400 samples) means noisy val loss curves
        * ReduceLROnPlateau adapts to actual observed plateaus rather than a fixed schedule
        * Complements early stopping naturally (LR halves first, then we stop if still stuck)
    - Early stopping: patience=4 epochs on val_loss (higher than scheduler patience=2
      so the LR reduction has a chance to rescue training before we give up)
    - Checkpoint: saved by best val_accuracy (consistent with Day 6 criterion)
    - Loss: BCEWithLogitsLoss (same as Day 6)

Run from project root:
    python model/train_finetune.py [--epochs N] [--batch-size B]
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
HEAD_ONLY_CKPT = CKPT_DIR / "day6_head_only_best.pth"
BEST_CKPT      = CKPT_DIR / "day8_finetuned_best.pth"

# ── Hyperparameters ───────────────────────────────────────────────────────── #
BACKBONE_LR    = 1e-5   # small LR to avoid destroying ImageNet features
HEAD_LR        = 1e-4   # 10x larger — head can converge faster
SCHED_FACTOR   = 0.5    # ReduceLROnPlateau: halve LR on plateau
SCHED_PATIENCE = 2      # LR reduction trigger: 2 epochs no val_loss improvement
ES_PATIENCE    = 4      # Early stopping: 4 epochs no val_loss improvement
MIN_LR         = 1e-7   # Floor for LR reduction


# ── Model ─────────────────────────────────────────────────────────────────── #
def build_and_load_model(device: torch.device) -> nn.Module:
    """
    Reconstruct EfficientNet-B0 with custom head, load Day 6 checkpoint,
    then UNFREEZE all layers for full fine-tuning.
    """
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)

    print(f"Loading head-only checkpoint: {HEAD_ONLY_CKPT}")
    ckpt = torch.load(HEAD_ONLY_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded from epoch {ckpt['epoch']} "
          f"(val_acc={ckpt['val_acc']:.4f}, val_loss={ckpt['val_loss']:.4f})")

    # Unfreeze ALL layers for end-to-end fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}  (entire network unfrozen)")

    return model.to(device)


def make_param_groups(model: nn.Module):
    """
    Differential learning rates:
      backbone → BACKBONE_LR
      head (classifier) → HEAD_LR
    """
    head_ids = {id(p) for p in model.classifier.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params     = [p for p in model.classifier.parameters()]

    return [
        {"params": backbone_params, "lr": BACKBONE_LR, "name": "backbone"},
        {"params": head_params,     "lr": HEAD_LR,     "name": "head"},
    ]


# ── DataLoaders ───────────────────────────────────────────────────────────── #
def make_loaders(batch_size: int):
    def _loader(split, transform, shuffle):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        ds = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _loader("train", get_train_transforms(),     shuffle=True),
        _loader("val",   get_val_test_transforms(),  shuffle=False),
    )


# ── Epoch helpers ─────────────────────────────────────────────────────────── #
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images)
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
    parser = argparse.ArgumentParser(description="Day 8: Full EfficientNet-B0 fine-tuning")
    parser.add_argument("--epochs",     type=int, default=30, help="Max epochs (early stopping may terminate sooner)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: max_epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        backbone_lr={BACKBONE_LR}, head_lr={HEAD_LR}")
    print(f"        scheduler=ReduceLROnPlateau(factor={SCHED_FACTOR}, patience={SCHED_PATIENCE})")
    print(f"        early_stopping patience={ES_PATIENCE}\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not HEAD_ONLY_CKPT.exists():
        print(f"ERROR: Head-only checkpoint not found at {HEAD_ONLY_CKPT}")
        sys.exit(1)

    model     = build_and_load_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(make_param_groups(model))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=MIN_LR
    )

    train_loader, val_loader = make_loaders(args.batch_size)
    print(f"\nDataset: train={len(train_loader.dataset)} | val={len(val_loader.dataset)}\n")

    # ── Training state ─────────────────────────────────────────────────── #
    best_val_acc   = 0.0
    best_val_loss  = float("inf")
    best_epoch     = 0
    es_counter     = 0           # epochs since last val_loss improvement
    log_rows       = []

    header = (f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7} | "
              f"{'BB LR':>8} | {'Hd LR':>8} | {'Time':>6}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        elapsed = time.time() - t0

        # Current LRs for logging
        bb_lr = optimizer.param_groups[0]["lr"]
        hd_lr = optimizer.param_groups[1]["lr"]

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%} | "
              f"{bb_lr:>8.2e} | {hd_lr:>8.2e} | {elapsed:>5.1f}s")

        # ── Scheduler step ─────────────────────────────────────────────── #
        scheduler.step(val_loss)

        # ── Best checkpoint ────────────────────────────────────────────── #
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":              val_acc,
                "val_loss":             val_loss,
                "config": {
                    "backbone_lr": BACKBONE_LR,
                    "head_lr":     HEAD_LR,
                    "batch_size":  args.batch_size,
                },
            }, BEST_CKPT)
            print(f"         -> Best checkpoint saved (val_acc={val_acc:.4f}, epoch={epoch})")

        # ── Early stopping on val_loss ──────────────────────────────────── #
        if val_loss < best_val_loss or is_best:
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= ES_PATIENCE:
            print(f"\n[Early Stopping] No val_loss improvement for {ES_PATIENCE} epochs. "
                  f"Stopping at epoch {epoch}.")
            break

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "backbone_lr": bb_lr,
            "head_lr":    hd_lr,
            "elapsed_s":  round(elapsed, 1),
        })

    # Append last epoch if not already added (when early stopping fires after append)
    if not log_rows or log_rows[-1]["epoch"] != epoch:
        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "backbone_lr": bb_lr,
            "head_lr":    hd_lr,
            "elapsed_s":  round(elapsed, 1),
        })

    # ── Write CSV log ──────────────────────────────────────────────────────── #
    log_path = LOG_DIR / "day8_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining log saved: {log_path}")
    print(f"Best checkpoint:    epoch={best_epoch}, val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
    print(f"Checkpoint path:    {BEST_CKPT}")


if __name__ == "__main__":
    main()
