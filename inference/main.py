import os
import sys
import time
import base64
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.face_extraction import extract_face
from data.dataset import get_val_test_transforms
from model.evaluate_fusion import build_model_from_ckpt
from model.train_fusion import FUSION_CKPT, FREQ_NPZ
from model.frequency_features import extract_radial_profile
from model.gradcam_viz import generate_cam

# App Setup
app = FastAPI(title="Deepfake Detector Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL = None
DEVICE = None
TRANSFORM = None
FREQ_MEAN = None
FREQ_STD = None
TARGET_LAYERS = None

class FusionCamWrapper(nn.Module):
    """Wrapper to pass only img_tensor to GradCAM while fixing freq_tensor"""
    def __init__(self, model, freq_tensor):
        super().__init__()
        self.model = model
        self.freq_tensor = freq_tensor
    def forward(self, x):
        return self.model(x, self.freq_tensor)

@app.on_event("startup")
def load_resources():
    global MODEL, DEVICE, TRANSFORM, FREQ_MEAN, FREQ_STD, TARGET_LAYERS
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # Save memory on Render free tier
    print(f"Loading fusion model onto {DEVICE}...")
    
    MODEL = build_model_from_ckpt(FUSION_CKPT, DEVICE)
    MODEL.eval()
    
    # Enable gradients on CNN for Grad-CAM
    for param in MODEL.cnn.parameters():
        param.requires_grad = True
        
    TARGET_LAYERS = [MODEL.cnn.conv_head] 
    
    TRANSFORM = get_val_test_transforms()
    
    print(f"Loading frequency normalization stats from {FREQ_NPZ}")
    npz = np.load(FREQ_NPZ)
    X_test = npz["X_test"].astype(np.float32)
    FREQ_MEAN = X_test.mean(axis=0)
    FREQ_STD = X_test.std(axis=0) + 1e-8
    print("Resources loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "day16_fusion_best.pth", "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Limit is 10MB.")
        
    temp_dir = ROOT / "inference" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_in = temp_dir / "temp_in.jpg"
    temp_face = temp_dir / "temp_face.jpg"
    
    with open(temp_in, "wb") as f:
        f.write(contents)
        
    try:
        # Extract face
        success = extract_face(str(temp_in), str(temp_face))
        if not success:
            raise HTTPException(status_code=400, detail="No face detected in the image or failed to process.")
            
        t0 = time.time()
        
        face_img = Image.open(temp_face).convert("RGB")
        img_tensor = TRANSFORM(face_img).unsqueeze(0).to(DEVICE)
        
        orig_img_np = np.array(face_img.resize((224, 224))) / 255.0
        
        freq_raw = extract_radial_profile(str(temp_face))
        freq_norm = (freq_raw.astype(np.float32) - FREQ_MEAN) / FREQ_STD
        freq_tensor = torch.tensor(freq_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = MODEL(img_tensor, freq_tensor)
            prob = torch.sigmoid(logit).item()
            pred_label = "fake" if prob > 0.5 else "real"
            
        t1 = time.time()
        inference_time_ms = (t1 - t0) * 1000
        
        # Grad-CAM overlay
        # We use torch.enable_grad() since we're in a FastAPI context where gradients might be disabled
        with torch.enable_grad():
            cam_wrapper = FusionCamWrapper(MODEL, freq_tensor)
            overlay, _ = generate_cam(cam_wrapper, TARGET_LAYERS, img_tensor, orig_img_np)
            
        # Convert overlay to base64
        overlay_img = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="JPEG", quality=85)
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "label": pred_label,
            "confidence": prob if pred_label == "fake" else 1 - prob,
            "probability_fake": prob,
            "inference_time_ms": round(inference_time_ms, 2),
            "heatmap_base64": overlay_b64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_in.exists():
            temp_in.unlink()
        if temp_face.exists():
            temp_face.unlink()
