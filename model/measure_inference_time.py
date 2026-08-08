"""
model/measure_inference_time.py
-------------------------------
Utility to benchmark inference latency (batch = 1) for the two model variants:
* CNN‑only checkpoint (`day32_finetuned_converged.pth`)
* Fusion checkpoint (`day32_fusion_converged.pth`)
The script reports the average time per image (ms) over the full test set.

Usage example:
    python model/measure_inference_time.py --model cnn   --ckpt model/checkpoints/day32_finetuned_converged.pth
    python model/measure_inference_time.py --model fusion --ckpt model/checkpoints/day32_fusion_converged.pth
"""

import argparse
import time
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np

class FusionDataset(Dataset):
    LABEL_MAP = {"real": 0, "fake": 1}
    def __init__(self, df, faces_dir, freq_features, transform=None):
        self.df = df.reset_index(drop=True)
        self.faces_dir = Path(faces_dir)
        self.transform = transform
        freq = freq_features.astype(np.float32)
        mean = freq.mean(axis=0); std = freq.std(axis=0) + 1e-8
        self.freq_norm = (freq - mean) / std
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.faces_dir / row["label"] / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        freq = torch.tensor(self.freq_norm[idx], dtype=torch.float32)
        label = torch.tensor(self.LABEL_MAP[row["label"]], dtype=torch.long)
        return image, freq, label

import timm

# Project‑relative imports
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from data.dataset import (
    DeepfakeFaceDataset,
    FusionDataset,
    get_val_test_transforms,
)

def load_cnn_ckpt(ckpt_path, device):
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def load_fusion_ckpt(fusion_ckpt_path, cnn_ckpt_path, device):
    # Load backbone from the CNN checkpoint (frozen)
    backbone = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, 1)
    cnn_ckpt = torch.load(cnn_ckpt_path, map_location=device)
    backbone.load_state_dict(cnn_ckpt["model_state_dict"])
    backbone.classifier = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad = False

    # Build fusion model (same definition as evaluate_day32_fusion)
    class FreqBranch(nn.Module):
        def __init__(self, in_dim=16, hidden=32, out_dim=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, out_dim), nn.ReLU(),
            )
        def forward(self, x):
            return self.net(x)

    class FusionModel(nn.Module):
        def __init__(self, cnn_backbone):
            super().__init__()
            self.cnn = cnn_backbone
            self.freq_branch = FreqBranch()
            self.head = nn.Linear(1280 + 16, 1)
        def forward(self, images, freq_feat):
            return self.head(torch.cat([self.cnn(images), self.freq_branch(freq_feat)], dim=1))

    model = FusionModel(backbone).to(device)
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location=device)
    model.load_state_dict(fusion_ckpt["model_state_dict"])
    model.eval()
    return model

def benchmark_cnn(ckpt_path):
    device = torch.device("cpu")
    model = load_cnn_ckpt(ckpt_path, device)
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    test_ds = DeepfakeFaceDataset(test_df, ROOT / "data" / "faces_extracted", transform=get_val_test_transforms())
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    times = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            t0 = time.time()
            _ = model(images)
            times.append((time.time() - t0) * 1000)  # ms
    return np.mean(times)

def benchmark_fusion(fusion_ckpt_path, cnn_ckpt_path):
    device = torch.device("cpu")
    model = load_fusion_ckpt(fusion_ckpt_path, cnn_ckpt_path, device)
    # Load frequency features
    freq_npz = np.load(ROOT / "data" / "frequency_features.npz")
    X_test = freq_npz["X_test"]
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    test_ds = FusionDataset(test_df, ROOT / "data" / "faces_extracted", X_test, transform=get_val_test_transforms())
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    times = []
    with torch.no_grad():
        for images, freq, _ in loader:
            images = images.to(device)
            freq = freq.to(device)
            t0 = time.time()
            _ = model(images, freq)
            times.append((time.time() - t0) * 1000)
    return np.mean(times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference time (batch=1) for CNN or Fusion model.")
    parser.add_argument("--model", choices=["cnn", "fusion"], required=True, help="Which model to benchmark.")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--cnn_ckpt", help="Path to CNN backbone checkpoint (required for fusion).")
    args = parser.parse_args()
    if args.model == "cnn":
        avg_ms = benchmark_cnn(Path(args.ckpt))
        print(f"CNN‑only average inference time (batch=1): {avg_ms:.2f} ms/image")
    else:
        if not args.cnn_ckpt:
            raise ValueError("--cnn_ckpt is required for fusion benchmarking")
        avg_ms = benchmark_fusion(Path(args.ckpt), Path(args.cnn_ckpt))
        print(f"Fusion average inference time (batch=1): {avg_ms:.2f} ms/image")
