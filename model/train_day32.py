"""
model/train_day32.py
---------------------
Day 32: Resume fine-tuning from day32_finetuned_converged.pth (epoch 6, best so far)
and run to TRUE convergence — early stopping patience=4 on val_loss.

Key differences from train_finetune.py:
  - Resumes from day32_finetuned_converged.pth (not day6_head_only_best.pth)
  - Saves per-epoch checkpoints (day32_epoch_N.pth) so progress is never lost
  - Saves best-so-far as day32_finetuned_converged.pth (overwrites)
  - Saves "latest" as day32_finetuned_latest.pth every epoch (for resume)
  - Writes live training log to results/logs/day32_training_log.csv
  - Max epochs=50, early stopping patience=4 on val_loss

Run from project root:
    python model/train_day32.py [--resume] [--max-epochs N] [--batch-size B]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_train_transforms, get_val_test_transforms

# ── Constants ─────────────────────────────────────────────────────────────── #
SPLITS_DIR      = ROOT / "data" / "splits"
FACES_DIR       = ROOT / "data" / "faces_extracted"
CKPT_DIR        = ROOT / "model" / "checkpoints"
LOG_DIR         = ROOT / "results" / "logs"

RESUME_CKPT     = CKPT_DIR / "day32_finetuned_latest.pth"      # resume from latest to avoid losing progress
BEST_CKPT       = CKPT_DIR / "day32_finetuned_converged.pth"   # overwrite with new best
LATEST_CKPT     = CKPT_DIR / "day32_finetuned_latest.pth"      # always last epoch

# Hyperparameters — identical to Day 8
BACKBONE_LR    = 1e-5
HEAD_LR        = 1e-4
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 2
ES_PATIENCE    = 4
MIN_LR         = 1e-7


def build_fresh_model(device):
    """Build EfficientNet-B0 with trainable classifier head."""
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    for param in model.parameters():
        param.requires_grad = True
    return model.to(device)


def make_param_groups(model):
    head_ids = {id(p) for p in model.classifier.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = list(model.classifier.parameters())
    return [
        {"params": backbone_params, "lr": BACKBONE_LR, "name": "backbone"},
        {"params": head_params,     "lr": HEAD_LR,     "name": "head"},
    ]


def make_loaders(batch_size):
    def _loader(split, transform, shuffle):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        ds = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return (
        _loader("train", get_train_transforms(), shuffle=True),
        _loader("val",   get_val_test_transforms(), shuffle=False),
    )


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
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


def save_ckpt(path, model, optimizer, epoch, val_acc, val_loss, extra=None):
    d = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":              val_acc,
        "val_loss":             val_loss,
        "config": {
            "backbone_lr": BACKBONE_LR,
            "head_lr":     HEAD_LR,
        },
    }
    if extra:
        d.update(extra)
    torch.save(d, path)


def main():
    parser = argparse.ArgumentParser(description="Day 32: Fine-tune to full convergence")
    parser.add_argument("--max-epochs",  type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--no-resume",   action="store_true",
                        help="Start fresh from day6 head-only ckpt instead of resuming")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: max_epochs={args.max_epochs}, batch_size={args.batch_size}")
    print(f"        backbone_lr={BACKBONE_LR}, head_lr={HEAD_LR}")
    print(f"        ES_PATIENCE={ES_PATIENCE}, SCHED_PATIENCE={SCHED_PATIENCE}\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model = build_fresh_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(make_param_groups(model))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=MIN_LR
    )

    # ── Resume ──────────────────────────────────────────────────────────────── #
    start_epoch = 1
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    es_counter    = 0
    log_rows      = []

    resume_path = RESUME_CKPT if not args.no_resume else None
    if resume_path and resume_path.exists():
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch    = ckpt["epoch"] + 1
        
        if BEST_CKPT.exists():
            best_ckpt = torch.load(BEST_CKPT, map_location=device)
            best_val_acc  = best_ckpt["val_acc"]
            best_val_loss = best_ckpt["val_loss"]
            best_epoch    = best_ckpt["epoch"]
            print(f"  True best loaded from BEST_CKPT: epoch={best_epoch}, val_acc={best_val_acc:.4f}")
        else:
            best_val_acc   = ckpt["val_acc"]
            best_val_loss  = ckpt["val_loss"]
            best_epoch     = ckpt["epoch"]
            
        print(f"  Resumed: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}, val_loss={ckpt['val_loss']:.4f}")
        print(f"  Will continue from epoch {start_epoch}\n")
    else:
        # Start from day6 head-only
        head_only = CKPT_DIR / "day6_head_only_best.pth"
        if head_only.exists():
            print(f"Loading head-only ckpt: {head_only}")
            ckpt = torch.load(head_only, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            for p in model.parameters():
                p.requires_grad = True
            print(f"  Loaded from epoch {ckpt['epoch']}")
        print("Starting from scratch (no resume checkpoint found)\n")

    train_loader, val_loader = make_loaders(args.batch_size)
    print(f"Dataset: train={len(train_loader.dataset)} | val={len(val_loader.dataset)}\n")

    header = (f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7} | "
              f"{'BB LR':>8} | {'Hd LR':>8} | {'Time':>6} | {'ES':>3}")
    print(header)
    print("-" * len(header))

    epoch = start_epoch - 1  # guard for empty loop
    train_loss = val_loss = train_acc = val_acc = 0.0

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        elapsed = time.time() - t0
        bb_lr = optimizer.param_groups[0]["lr"]
        hd_lr = optimizer.param_groups[1]["lr"]

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%} | "
              f"{bb_lr:>8.2e} | {hd_lr:>8.2e} | {elapsed:>5.1f}s | {es_counter:>3}")

        scheduler.step(val_loss)

        # ── Checkpoint every epoch ──────────────────────────────────────────── #
        epoch_ckpt = CKPT_DIR / f"day32_epoch_{epoch:02d}.pth"
        save_ckpt(epoch_ckpt, model, optimizer, epoch, val_acc, val_loss)
        save_ckpt(LATEST_CKPT, model, optimizer, epoch, val_acc, val_loss)

        # ── Best checkpoint ─────────────────────────────────────────────────── #
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            es_counter    = 0
            save_ckpt(BEST_CKPT, model, optimizer, epoch, val_acc, val_loss,
                      extra={"best_epoch": epoch})
            print(f"         -> Best checkpoint saved (val_acc={val_acc:.4f}, epoch={epoch})")
        else:
            # Only reset es_counter on val_loss improvement (even if acc same)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                es_counter = 0
            else:
                es_counter += 1

        row = {
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "train_acc":   round(train_acc,  6),
            "val_loss":    round(val_loss,   6),
            "val_acc":     round(val_acc,    6),
            "backbone_lr": bb_lr,
            "head_lr":     hd_lr,
            "elapsed_s":   round(elapsed, 1),
            "es_counter":  es_counter,
        }
        log_rows.append(row)

        # Write CSV incrementally (so we don't lose logs on crash)
        log_path = LOG_DIR / "day32_training_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(log_rows)

        if es_counter >= ES_PATIENCE:
            print(f"\n[Early Stopping] No val_loss improvement for {ES_PATIENCE} epochs. "
                  f"Stopping at epoch {epoch}.")
            break

    print(f"\nTraining complete.")
    print(f"Best: epoch={best_epoch}, val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    main()
