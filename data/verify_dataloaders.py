"""
data/verify_dataloaders.py
--------------------------
Wires up PyTorch DataLoaders for train/val/test splits and confirms:
  - Batch shapes are correct  (B, 3, 224, 224)
  - Labels are correct dtypes and in {0, 1}
  - Label balance within a single batch (informational only -- not guaranteed per-batch)

Run from project root:
    python data/verify_dataloaders.py
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import (
    DeepfakeFaceDataset,
    get_train_transforms,
    get_val_test_transforms,
)

SPLITS_DIR = os.path.join("data", "splits")
FACES_DIR  = os.path.join("data", "faces_extracted")
BATCH_SIZE = 32


def make_loader(csv_path: str, transform, shuffle: bool) -> DataLoader:
    df = pd.read_csv(csv_path)
    dataset = DeepfakeFaceDataset(df, FACES_DIR, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def main():
    train_loader = make_loader(
        os.path.join(SPLITS_DIR, "train.csv"),
        get_train_transforms(),
        shuffle=True,
    )
    val_loader = make_loader(
        os.path.join(SPLITS_DIR, "val.csv"),
        get_val_test_transforms(),
        shuffle=False,
    )
    test_loader = make_loader(
        os.path.join(SPLITS_DIR, "test.csv"),
        get_val_test_transforms(),
        shuffle=False,
    )

    print(f"DataLoader sizes: train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | test={len(test_loader.dataset)}")
    print()

    for split_name, loader in [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]:
        images, labels = next(iter(loader))
        real_n = (labels == 0).sum().item()
        fake_n = (labels == 1).sum().item()
        print(f"[{split_name}] batch shape: {tuple(images.shape)}  dtype={images.dtype}")
        print(f"        labels: {labels.tolist()[:16]}...")
        print(f"        label dtype: {labels.dtype}")
        print(f"        unique labels in batch: {labels.unique().tolist()}")
        print(f"        real(0)={real_n}  fake(1)={fake_n}  in this batch of {len(labels)}")
        assert images.shape[1:] == (3, 224, 224), f"Unexpected image shape: {images.shape}"
        assert labels.dtype == torch.long, f"Unexpected label dtype: {labels.dtype}"
        assert set(labels.unique().tolist()).issubset({0, 1}), "Labels outside {0,1} found!"
        print(f"        [OK] Assertions passed.\n")

    print("All DataLoaders verified successfully.")


if __name__ == "__main__":
    main()
