import os
import sys
import time
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import get_val_test_transforms
from model.evaluate_fusion import build_model_from_ckpt
from model.train_fusion import FUSION_CKPT, FREQ_NPZ
from model.frequency_features import extract_radial_profile

# App Setup
app = FastAPI(title="Deepfake Detector Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": "Too many requests. The limit is 10 requests per minute per IP address. Please try again later."}
    )

# Global state
MODEL = None
DEVICE = None
TRANSFORM = None
FREQ_MEAN = None
FREQ_STD = None


def haar_extract_face(img_bytes: bytes) -> Image.Image | None:
    """
    Lightweight face extraction using only OpenCV Haar Cascades.
    Avoids importing MTCNN/TensorFlow entirely (saves ~400MB RAM).
    Falls back to full-image centre-crop if no face is detected.
    """
    import cv2
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    img_h, img_w = img_bgr.shape[:2]

    # Try Haar Cascade face detection
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_crop = None

    if os.path.exists(cascade_path):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            largest = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = largest
            margin = 0.2
            mx, my = int(w * margin), int(h * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(img_w, x + w + mx)
            y2 = min(img_h, y + h + my)
            face_crop = img_bgr[y1:y2, x1:x2]

    if face_crop is None or face_crop.size == 0:
        return None

    face_bgr = cv2.resize(face_crop, (224, 224))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def gradcam_overlay(model: nn.Module, img_tensor: torch.Tensor,
                    freq_tensor: torch.Tensor, orig_np: np.ndarray) -> str:
    """
    Lightweight manual Grad-CAM — no external library needed.
    Uses full model forward pass for correct scalar loss.
    Returns base64-encoded JPEG of the overlay.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.cnn.conv_head
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        img_in = img_tensor.clone().detach().requires_grad_(False)
        freq_in = freq_tensor.clone().detach()
        logit = model(img_in, freq_in)
        model.zero_grad()
        logit.backward()

        if not activations or not gradients:
            return None

        acts = activations[0].detach()   # [1, C, H, W]
        grads = gradients[0].detach()    # [1, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        import cv2
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig_uint8 = np.uint8(orig_np * 255)
        overlay = (0.5 * orig_uint8 + 0.5 * heatmap_rgb).astype(np.uint8)

        overlay_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    finally:
        fh.remove()
        bh.remove()


@app.on_event("startup")
def load_resources():
    global MODEL, DEVICE, TRANSFORM, FREQ_MEAN, FREQ_STD

    torch.set_num_threads(1)
    DEVICE = torch.device("cpu")
    print("Loading fusion model onto cpu...")

    MODEL = build_model_from_ckpt(FUSION_CKPT, DEVICE)
    MODEL.eval()

    TRANSFORM = get_val_test_transforms()

    print(f"Loading frequency normalization stats from {FREQ_NPZ}")
    npz = np.load(FREQ_NPZ)
    X_test = npz["X_test"].astype(np.float32)
    FREQ_MEAN = X_test.mean(axis=0)
    FREQ_STD = X_test.std(axis=0) + 1e-8
    print("Resources loaded successfully.")


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "day16_fusion_best.pth", "device": "cpu", "version": "day31-v2"}

@app.get("/ip")
def get_ip(request: Request):
    return {
        "client_host": request.client.host if request.client else None,
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "slowapi_ip": get_remote_address(request)
    }

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Limit is 10MB.")

    try:
        # Extract face (Haar only — no MTCNN/TF needed)
        face_pil = haar_extract_face(contents)
        if face_pil is None:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image. Please upload an image containing a clear, visible face.")

        t0 = time.time()

        img_tensor = TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)
        orig_np = np.array(face_pil.resize((224, 224))) / 255.0

        # Save face to temp file for FFT extraction (requires a path)
        tmp_path = ROOT / "inference" / "tmp" / "face_tmp.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        face_pil.save(str(tmp_path), format="JPEG")

        freq_raw = extract_radial_profile(str(tmp_path))
        freq_norm = (freq_raw.astype(np.float32) - FREQ_MEAN) / FREQ_STD
        freq_tensor = torch.tensor(freq_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = MODEL(img_tensor, freq_tensor)
            prob = torch.sigmoid(logit).item()
            pred_label = "fake" if prob > 0.5 else "real"

        t1 = time.time()
        inference_ms = round((t1 - t0) * 1000, 2)

        # Grad-CAM overlay (manual, no external lib)
        try:
            with torch.enable_grad():
                # Temporarily re-enable grads on CNN params for Grad-CAM
                for p in MODEL.cnn.parameters():
                    p.requires_grad_(True)
                heatmap_b64 = gradcam_overlay(MODEL, img_tensor.clone(), freq_tensor.clone(), orig_np)
                for p in MODEL.cnn.parameters():
                    p.requires_grad_(False)
        except Exception:
            heatmap_b64 = None

        return {
            "label": pred_label,
            "confidence": prob if pred_label == "fake" else 1 - prob,
            "probability_fake": prob,
            "inference_time_ms": inference_ms,
            "heatmap_base64": heatmap_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
