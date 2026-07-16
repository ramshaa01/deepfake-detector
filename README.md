# Deepfake / Manipulated Media Detector

## Problem Statement
With the rise of generative AI, the risk of misinformation, fraud, and impersonation on platforms handling User-Generated Content (UGC) has significantly increased. This project aims to build a robust system that detects manipulated facial imagery and video with interpretable, quantified confidence.

## Tech Stack
- **ML Core**: PyTorch, EfficientNet-B0, OpenCV, MTCNN (face extraction), Grad-CAM (interpretability)
  - *Note: Python 3.10 or 3.11 is explicitly required for this project to ensure compatibility with the MTCNN/TensorFlow stack.*
- **Backend**: FastAPI
- **Frontend**: React, Tailwind CSS, Recharts
- **Deployment**: HuggingFace Spaces/Render (Backend), Vercel (Frontend)

## Repo Structure
```text
deepfake-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_dataset.py
│   └── raw_samples/
├── model/
├── inference/
├── frontend/
├── notebooks/
└── results/
```

## Results
*(Placeholder: To be filled in later with confusion matrix, ROC-AUC, per-manipulation accuracy, and robustness table)*

**Status: Day 2 of 30 — face extraction pipeline built**
