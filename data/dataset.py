"""
data/dataset.py
---------------
PyTorch Dataset class for the Deepfake / Manipulated Media Detector project.

Label convention (used consistently throughout this project):
    0 = REAL (authentic face)
    1 = FAKE (manipulated / deepfake face)

Dataset source: Kaggle Deepfake Detection Challenge subset
Face crops: 224×224 px, extracted via MTCNN, stored in data/faces_extracted/
Labels:     tracked in data/faces_extracted/metadata.csv
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------------------------------- #
#  ImageNet statistics — EfficientNet-B0 is ImageNet-pretrained               #
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """
    Light augmentation pipeline for training.

    NOTE: We deliberately avoid aggressive blur, JPEG compression, or
    frequency-domain augmentations because those would destroy the subtle
    forensic artifacts (blocking artefacts, GAN fingerprints, blending
    boundaries) that EfficientNet-B0 needs to learn to detect.  Only
    geometry/colour augmentations that preserve pixel authenticity are used.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),                          # Confirm 224×224
        transforms.RandomHorizontalFlip(p=0.5),                # Mirror faces
        transforms.RandomRotation(degrees=10),                  # ±10° rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle lighting
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for validation and test sets — no augmentation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DeepfakeFaceDataset(Dataset):
    """
    PyTorch Dataset for deepfake face classification.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        A slice of the metadata CSV already filtered to the desired split.
        Expected columns: filename, label, source_path, extraction_status.
    faces_dir : str | Path
        Root directory that contains label-named subdirectories
        (e.g. data/faces_extracted/real/ and data/faces_extracted/fake/).
    transform : callable, optional
        Torchvision transform to apply to each image.

    Returns (per __getitem__)
    -------
    image : torch.Tensor  shape (3, 224, 224), float32
    label : torch.Tensor  scalar int64;  0 = real, 1 = fake
    """

    LABEL_MAP = {"real": 0, "fake": 1}  # 0 = real, 1 = fake

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        faces_dir: str | Path,
        transform=None,
    ):
        # Filter to successful extractions only
        self.df = (
            metadata_df[metadata_df["extraction_status"] == "success"]
            .reset_index(drop=True)
        )
        self.faces_dir = Path(faces_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Build image path: faces_dir / label / filename
        img_path = self.faces_dir / row["label"] / row["filename"]

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Face crop not found at expected path: {img_path}\n"
                f"  source_path in CSV: {row['source_path']}"
            )

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.LABEL_MAP[row["label"]], dtype=torch.long)
        return image, label
