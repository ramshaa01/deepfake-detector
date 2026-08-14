"""
model/train_day37_seed7.py
--------------------------
Day 37 Part A: Train seed=7 to full convergence to measure variance.

This script runs both stages in sequence:
Stage 1: Head-only training (10 epochs, lr=1e-3)
Stage 2: Full fine-tuning (patience=4 early stopping, differential LRs)
"""

import os
import sys
import argparse
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

SPLITS_DIR      = ROOT / "data" / "splits"
FACES_DIR       = ROOT / "data" / "faces_extracted"
CKPT_DIR        = ROOT / "model" / "checkpoints"
LOG_DIR         = ROOT / "results" / "logs"

SEED = 7
HEAD_ONLY_CKPT  = CKPT_DIR / "day37_seed7_head_only_best.pth"
BEST_CKPT       = CKPT_DIR / "day37_seed7_best.pth"
LATEST_CKPT     = CKPT_DIR / "day37_seed7_latest.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(batch_size):
    def _loader(split, transform, shuffle):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        ds = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
        # Apply generator for reproducibility in DataLoader shuffle
        generator = torch.Generator()
        generator.manual_seed(SEED)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator if shuffle else None)
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


def run_stage1(device, train_loader, val_loader):
    print("\n" + "="*50)
    print("STAGE 1: HEAD-ONLY TRAINING")
    print("="*50)
    
    model = timm.create_model("efficientnet_b0", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    epochs = 10
    best_val_acc = 0.0
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:>2} | Tr Loss: {train_loss:.4f} | Tr Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | {elapsed:.1f}s")
        
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc}, HEAD_ONLY_CKPT)
            print(f"  -> Best head-only ckpt saved (val_acc={val_acc:.4f})")
    print(f"Stage 1 complete. Best val_acc: {best_val_acc:.4f}\n")


def run_stage2(device, train_loader, val_loader):
    print("\n" + "="*50)
    print("STAGE 2: FULL FINE-TUNING")
    print("="*50)
    
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    
    ckpt = torch.load(HEAD_ONLY_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)
    
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-4
    head_ids = {id(p) for p in model.classifier.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = list(model.classifier.parameters())
    param_groups = [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ]
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7)
    
    max_epochs = 50
    ES_PATIENCE = 4
    
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    es_counter    = 0
    
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0
        
        bb_lr = optimizer.param_groups[0]["lr"]
        hd_lr = optimizer.param_groups[1]["lr"]
        
        print(f"Epoch {epoch:>2} | Tr Loss: {train_loss:.4f} | Tr Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | BB LR: {bb_lr:.1e} | Hd LR: {hd_lr:.1e} | ES: {es_counter} | {elapsed:.1f}s")
        
        scheduler.step(val_loss)
        
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            es_counter    = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, BEST_CKPT)
            print(f"  -> Best fine-tune ckpt saved (val_acc={val_acc:.4f})")
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                es_counter = 0
            else:
                es_counter += 1
                
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, LATEST_CKPT)
        
        if es_counter >= ES_PATIENCE:
            print(f"\n[Early Stopping] No val_loss improvement for {ES_PATIENCE} epochs. Stopping at epoch {epoch}.")
            break

    print(f"\nStage 2 complete. Best epoch={best_epoch}, val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Seed {SEED} training pipeline...")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, val_loader = make_loaders(batch_size=32)
    
    run_stage1(device, train_loader, val_loader)
    run_stage2(device, train_loader, val_loader)

if __name__ == "__main__":
    main()
