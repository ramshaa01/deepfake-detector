"""
model/train_day37_seed123.py
----------------------------
Day 37 Part B: Train seed=123 to full convergence for multi-seed variance reporting.

Fixes from Part A:
  - sys.stdout.flush() after every print for real-time monitoring
  - Incremental CSV log written every epoch (results/logs/day37b_seed123_training_log.csv)
  - Run with: python -u model/train_day37_seed123.py 2>&1 | Tee-Object log.txt

Pipeline identical to Day 32 / Part A (seed=7):
  Stage 1: Head-only (10 epochs, lr=1e-3, pretrained backbone frozen)
  Stage 2: Full fine-tune (differential LRs 1e-5/1e-4, early stopping patience=4)
"""

import os
import sys
import csv
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset import DeepfakeFaceDataset, get_train_transforms, get_val_test_transforms

SPLITS_DIR  = ROOT / "data" / "splits"
FACES_DIR   = ROOT / "data" / "faces_extracted"
CKPT_DIR    = ROOT / "model" / "checkpoints"
LOG_DIR     = ROOT / "results" / "logs"

SEED = 123
HEAD_ONLY_CKPT = CKPT_DIR / "day37_seed123_head_only_best.pth"
BEST_CKPT      = CKPT_DIR / "day37_seed123_best.pth"
LATEST_CKPT    = CKPT_DIR / "day37_seed123_latest.pth"
LOG_CSV        = LOG_DIR  / "day37b_seed123_training_log.csv"

BACKBONE_LR    = 1e-5
HEAD_LR        = 1e-4
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 2
ES_PATIENCE    = 4
MIN_LR         = 1e-7
HEAD_EPOCHS    = 10
MAX_EPOCHS     = 50
BATCH_SIZE     = 32


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders():
    def _loader(split, transform, shuffle):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        ds = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
        generator = torch.Generator()
        generator.manual_seed(SEED)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
            generator=generator if shuffle else None
        )
    return (
        _loader("train", get_train_transforms(),    shuffle=True),
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


def write_csv(log_rows):
    """Append-write the full log so it's always up to date on disk."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)


# ── Stage 1: Head-only ────────────────────────────────────────────────────── #
def run_stage1(device, train_loader, val_loader):
    log("\n" + "="*60)
    log("STAGE 1: HEAD-ONLY TRAINING  (seed={})".format(SEED))
    log("="*60)

    model = timm.create_model("efficientnet_b0", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_val_acc  = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, HEAD_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        log(f"[S1] Ep {epoch:>2}/{HEAD_EPOCHS} | "
            f"Tr {tr_loss:.4f}/{tr_acc:.2%} | "
            f"Va {va_loss:.4f}/{va_acc:.2%} | {elapsed:.1f}s")

        is_best = (va_acc > best_val_acc) or (va_acc == best_val_acc and va_loss < best_val_loss)
        if is_best:
            best_val_acc  = va_acc
            best_val_loss = va_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": va_acc, "val_loss": va_loss}, HEAD_ONLY_CKPT)
            log(f"       -> Best head-only ckpt saved (val_acc={va_acc:.4f})")

    log(f"\nStage 1 done. Best val_acc={best_val_acc:.4f}\n")
    return best_val_acc


# ── Stage 2: Full fine-tune ───────────────────────────────────────────────── #
def run_stage2(device, train_loader, val_loader):
    log("="*60)
    log("STAGE 2: FULL FINE-TUNING  (seed={})".format(SEED))
    log("="*60)

    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)

    ckpt = torch.load(HEAD_ONLY_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)

    head_ids       = {id(p) for p in model.classifier.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params     = list(model.classifier.parameters())
    param_groups    = [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE, min_lr=MIN_LR
    )

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    es_counter    = 0
    log_rows      = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        bb_lr = optimizer.param_groups[0]["lr"]
        hd_lr = optimizer.param_groups[1]["lr"]

        log(f"[S2] Ep {epoch:>2}/{MAX_EPOCHS} | "
            f"Tr {tr_loss:.4f}/{tr_acc:.2%} | "
            f"Va {va_loss:.4f}/{va_acc:.2%} | "
            f"BB {bb_lr:.1e} Hd {hd_lr:.1e} | ES {es_counter}/{ES_PATIENCE} | {elapsed:.1f}s")

        scheduler.step(va_loss)

        is_best = (va_acc > best_val_acc) or (va_acc == best_val_acc and va_loss < best_val_loss)
        if is_best:
            best_val_acc  = va_acc
            best_val_loss = va_loss
            best_epoch    = epoch
            es_counter    = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":  va_acc,
                "val_loss": va_loss,
            }, BEST_CKPT)
            log(f"       -> Best ckpt saved (val_acc={va_acc:.4f}, epoch={epoch})")
        else:
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                es_counter = 0
            else:
                es_counter += 1

        # Latest-epoch ckpt (for crash recovery)
        torch.save({
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc":  va_acc,
            "val_loss": va_loss,
        }, LATEST_CKPT)

        log_rows.append({
            "stage": 2, "epoch": epoch,
            "tr_loss": round(tr_loss, 6), "tr_acc": round(tr_acc, 6),
            "va_loss": round(va_loss, 6), "va_acc": round(va_acc, 6),
            "backbone_lr": bb_lr, "head_lr": hd_lr,
            "es_counter": es_counter, "elapsed_s": round(elapsed, 1),
        })
        write_csv(log_rows)  # flush to disk after every epoch

        if es_counter >= ES_PATIENCE:
            log(f"\n[Early Stopping] val_loss no improvement for {ES_PATIENCE} epochs. "
                f"Stopping at epoch {epoch}.")
            break

    log(f"\nStage 2 done. Best: epoch={best_epoch}, val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
    return best_val_acc, best_val_loss, best_epoch


# ── Main ──────────────────────────────────────────────────────────────────── #
def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Day 37 Part B — Seed={SEED} training")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    train_loader, val_loader = make_loaders()
    log(f"Dataset: train={len(train_loader.dataset)} | val={len(val_loader.dataset)}\n")

    s1_best_acc = run_stage1(device, train_loader, val_loader)
    s2_best_acc, s2_best_loss, s2_best_epoch = run_stage2(device, train_loader, val_loader)

    log("\n" + "="*60)
    log("TRAINING COMPLETE")
    log(f"  Stage 1 best val_acc: {s1_best_acc:.4f}")
    log(f"  Stage 2 best val_acc: {s2_best_acc:.4f}  (epoch {s2_best_epoch})")
    log(f"  Best ckpt: {BEST_CKPT}")
    log(f"  Training log: {LOG_CSV}")
    log("="*60)


if __name__ == "__main__":
    main()
