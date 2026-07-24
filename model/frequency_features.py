"""
model/frequency_features.py
---------------------------
Day 15: FFT-based Frequency-Domain Features

This script:
1. Computes the 2D FFT of face crops.
2. Extracts a 1D radial energy distribution feature vector (16 bins).
3. Plots the average radial profile for Real vs Fake samples.
4. Extracts features for the entire dataset and saves them to data/frequency_features.npz.
5. Trains a standalone Logistic Regression classifier on these features and evaluates it.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FACES_DIR = ROOT / "data" / "faces_extracted"
SPLITS_DIR = ROOT / "data" / "splits"
FEATURES_OUT = ROOT / "data" / "frequency_features.npz"
PLOT_OUT = ROOT / "results" / "day15_fft_profiles.png"

BINS = 16

def extract_radial_profile(img_path, bins=BINS):
    """
    Reads an image, converts to grayscale, computes 2D FFT, and extracts
    the radial energy distribution (azimuthal average).
    Returns a 1D numpy array of length `bins`.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
        
    # Resize to a fixed size for consistent FFT dimensions (e.g., 256x256)
    img = cv2.resize(img, (256, 256))
    
    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Magnitude spectrum (log scale for numerical stability and visualization)
    magnitude = np.log(np.abs(fshift) + 1e-8)
    
    # Calculate the distance of each pixel from the center
    center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Max radius in the image
    r_max = np.max(r)
    
    # Create radial bins
    radial_profile = np.zeros(bins)
    bin_edges = np.linspace(0, r_max, bins + 1)
    
    for i in range(bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if np.any(mask):
            radial_profile[i] = np.mean(magnitude[mask])
        else:
            radial_profile[i] = 0
            
    return radial_profile


def process_split(split_df):
    """
    Extracts radial profiles for all images in a dataframe.
    """
    features = []
    labels = []
    for _, row in split_df.iterrows():
        path = FACES_DIR / row['label'] / row['filename']
        try:
            profile = extract_radial_profile(path)
            features.append(profile)
            labels.append(1 if row['label'] == 'fake' else 0)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return np.array(features), np.array(labels)


def main():
    print("Loading splits...")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    # 1. Visual Comparison on a sample
    print("Generating radial profile comparison plot on sample data...")
    sample_real = train_df[train_df['label'] == 'real'].sample(30, random_state=42)
    sample_fake = train_df[train_df['label'] == 'fake'].sample(30, random_state=42)
    
    real_profiles = [extract_radial_profile(FACES_DIR / 'real' / row['filename']) for _, row in sample_real.iterrows()]
    fake_profiles = [extract_radial_profile(FACES_DIR / 'fake' / row['filename']) for _, row in sample_fake.iterrows()]
    
    avg_real = np.mean(real_profiles, axis=0)
    avg_fake = np.mean(fake_profiles, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(BINS), avg_real, label='Real (Average of 30)', marker='o')
    plt.plot(range(BINS), avg_fake, label='StyleGAN2 Fake (Average of 30)', marker='s')
    plt.title('Average Radial Energy Distribution (FFT Magnitude)')
    plt.xlabel('Frequency Bin (Low to High Frequency)')
    plt.ylabel('Average Log Magnitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"Saved plot to {PLOT_OUT}")
    
    # 2. Extract for full dataset
    print("\nExtracting features for the full dataset (this may take a minute)...")
    X_train, y_train = process_split(train_df)
    X_val, y_val = process_split(val_df)
    X_test, y_test = process_split(test_df)
    
    # Save features
    np.savez(FEATURES_OUT, 
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print(f"Saved full feature matrices to {FEATURES_OUT}")
    print(f"Feature vector shape: {X_train.shape[1]} dimensions (each representing a radial frequency band)")
    
    # 3. Train Standalone Classifier (Logistic Regression)
    print("\nTraining Standalone Frequency-Only Classifier (Logistic Regression)...")
    
    # We train on train+val for the final model (or just train and evaluate on val, then test)
    # Let's train on train, validate on val (hyperparam tune if needed), then test.
    # We'll just use default LogisticRegression for a simple baseline.
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Eval on Test
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    print("\n--- TEST SET PERFORMANCE (Frequency-Only) ---")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(f"ROC-AUC:  {test_auc:.4f}")

if __name__ == "__main__":
    main()
