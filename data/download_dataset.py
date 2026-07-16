"""
Dataset Download Script for Deepfake/Manipulated Media Detector

Note on FaceForensics++:
The full FaceForensics++ dataset requires a signed usage agreement to be submitted
to the authors. It cannot be automatically downloaded without authorization.
If you have access, you can download it using the official script provided by the authors.

This script provides a fallback to download a subset of the Kaggle Deepfake Detection Challenge.

Prerequisites for Kaggle download:
1. Create a Kaggle account.
2. Go to your Account settings and create a New API Token.
3. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/Mac) 
   or `C:\\Users\\<Username>\\.kaggle\\kaggle.json` (Windows).
4. Install kaggle API: `pip install kaggle`
"""

import argparse
import os
import subprocess
import sys

def download_kaggle_subset(subset: str, output_dir: str):
    """
    Downloads a sample subset from Kaggle Deepfake Detection Challenge.
    Requires kaggle API to be configured.
    """
    print(f"Attempting to download Kaggle dataset subset: {subset}")
    
    # "140k Real and Fake Faces" — 70k FFHQ real faces + 70k StyleGAN fake faces.
    # Freely accessible via the Kaggle API (CC0 license, no rules acceptance required).
    # Dataset page: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    dataset_name = "xhlulu/140k-real-and-fake-faces"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use subprocess to call the kaggle CLI
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", dataset_name,
            "-p", output_dir,
            "--unzip"
        ], check=True)
        print(f"Successfully downloaded dataset to {output_dir}")
        print("Next step: Run the face_extraction.py script on the raw data.")
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please ensure kaggle API is installed (`pip install kaggle`) and configured.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset. Ensure your kaggle.json is set up correctly. Details: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download datasets for deepfake detection.")
    parser.add_argument("--source", type=str, choices=["kaggle"], default="kaggle",
                        help="Data source to download from (default: kaggle)")
    parser.add_argument("--subset", type=str, default="small",
                        help="Subset of the dataset to download (e.g., small, full)")
    parser.add_argument("--output_dir", type=str, default="data/raw_samples",
                        help="Output directory for raw samples")
    
    args = parser.parse_args()
    
    if args.source == "kaggle":
        download_kaggle_subset(args.subset, args.output_dir)
    else:
        print(f"Source {args.source} is not implemented.")

if __name__ == "__main__":
    main()
