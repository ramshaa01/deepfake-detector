# Deepfake / Manipulated Media Detector

## Problem Statement
With the rise of generative AI, the risk of misinformation, fraud, and impersonation on platforms handling User-Generated Content (UGC) has significantly increased. This project aims to build a robust system that detects manipulated facial imagery and video with **interpretable, quantified confidence**.

## Tech Stack
- **ML Core**: PyTorch, EfficientNet-B0 (transfer learning), OpenCV, MTCNN (face extraction), Grad-CAM (interpretability), FFT-based frequency features
  - *Python 3.11 required — needed for MTCNN/TensorFlow compatibility*
- **Backend**: FastAPI
- **Frontend**: React, Tailwind CSS, Recharts
- **Deployment**: HuggingFace Spaces / Render (Backend), Vercel (Frontend)

## Repo Structure
```
deepfake-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_dataset.py       # Kaggle dataset downloader
│   ├── face_extraction.py        # MTCNN + Haar Cascade face extractor
│   ├── batch_extract.py          # Balanced batch extraction (1000 real + 1000 fake)
│   ├── dataset.py                # PyTorch DeepfakeFaceDataset class + transforms
│   ├── make_splits.py            # Stratified 70/15/15 train/val/test split
│   ├── verify_dataloaders.py     # DataLoader batch shape/label assertions
│   ├── sanity_check.py           # Class balance report + sample grids + leakage check
│   ├── splits/
│   │   ├── train.csv             # 1,400 samples (700 real / 700 fake)
│   │   ├── val.csv               # 300 samples  (150 real / 150 fake)
│   │   └── test.csv              # 300 samples  (150 real / 150 fake)
│   └── faces_extracted/
│       └── metadata.csv          # Extraction metadata for all 2,000 face crops
├── model/                        # EfficientNet-B0 model (Day 6+)
├── inference/                    # FastAPI inference endpoint (Day 13+)
├── frontend/                     # React frontend (Day 20+)
├── notebooks/                    # EDA and experiment notebooks
└── results/
    ├── day2_notes.md
    ├── day2_sample_crops/
    ├── day3_notes.md
    ├── day3_class_balance_check.md
    ├── day4_notes.md
    ├── day4_train_sample_grid.png
    ├── day4_val_sample_grid.png
    └── day4_test_sample_grid.png
```

## Dataset
- **Source**: Kaggle Deepfake Detection Challenge subset ([xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces))
- **Extracted**: 2,000 balanced face crops (1,000 real, 1,000 fake) via MTCNN — 100% success rate
- **Split**: 1,400 train / 300 val / 300 test — stratified, 50/50 class balance in every split
- **Label convention**: `0` = real, `1` = fake

## Results
*(To be filled in after training — confusion matrix, ROC-AUC, per-class accuracy, Grad-CAM visualizations, robustness table)*

## Progress Log

| Day | Task | Status |
|---|---|---|
| 1 | Project scaffold, requirements, dataset downloader | ✅ Done |
| 2 | Face extraction pipeline (MTCNN + Haar Cascade fallback) | ✅ Done |
| 3 | Batch extraction (2,000 balanced crops), metadata CSV | ✅ Done |
| 4 | PyTorch Dataset class, stratified train/val/test splits, DataLoader verification | ✅ Done |
| 5 | EDA notebook, class analysis | ⏳ Upcoming |
| 6–12 | EfficientNet-B0 transfer learning + training | ⏳ Upcoming |
| 13–15 | Grad-CAM interpretability + FFT frequency features | ⏳ Upcoming |
| 16–19 | FastAPI backend + inference endpoint | ⏳ Upcoming |
| 20–24 | React + Recharts frontend | ⏳ Upcoming |
| 25–27 | Robustness testing (JPEG/blur/resize) | ⏳ Upcoming |
| 28–30 | Deployment + final metrics | ⏳ Upcoming |

## Setup

```bash
# Python 3.11 required
py -3.11 -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

# Download dataset (requires Kaggle API credentials)
python data/download_dataset.py --source kaggle --subset small

# Run face extraction (balanced 1000 real + 1000 fake)
python data/batch_extract.py

# Generate train/val/test splits
python data/make_splits.py

# Verify DataLoaders
python data/verify_dataloaders.py
```
