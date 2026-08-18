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
import timm

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


def gradcam_overlay(model: nn.Module, img_tensor: torch.Tensor, freq_tensor: torch.Tensor | None, orig_np: np.ndarray) -> str:
    """Generate a Grad‑CAM overlay for a CNN‑only model.

    The original fusion implementation required both image and frequency tensors.
    For the CNN‑only deployment we ignore ``freq_tensor`` and run a forward pass
    with only ``img_tensor``. The function still registers hooks on the CNN’s
    ``conv_head`` layer to capture activations and gradients, computes a heat‑map,
    blends it with the original image, and returns a base64‑encoded JPEG.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # The EfficientNet backbone stores its final conv block as ``conv_head``
    target_layer = model.conv_head if hasattr(model, "conv_head") else model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        img_in = img_tensor.clone().detach().requires_grad_(True)
        # Forward pass – ignore freq_tensor for CNN‑only model
        logit = model(img_in)
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
    global MODEL, DEVICE, TRANSFORM

    torch.set_num_threads(1)
    DEVICE = torch.device("cpu")
    print("Loading CNN-only model onto cpu...")

    # Load the converged CNN‑only checkpoint (day32_finetuned_converged.pth)
    ckpt_path = ROOT / "model" / "checkpoints" / "day32_finetuned_converged.pth"
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    MODEL = model

    TRANSFORM = get_val_test_transforms()
    print("Resources loaded successfully.")


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "day32_finetuned_converged.pth", "device": "cpu", "version": "day32-prod"}

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
        t_start = time.time()

        # Extract face using Haar Cascade (no MTCNN/TF needed)
        t_haar_start = time.time()
        face_pil = haar_extract_face(contents)
        t_haar_end = time.time()
        
        if face_pil is None:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image. Please upload an image containing a clear, visible face.")

        t_pre_start = time.time()
        img_tensor = TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)
        orig_np = np.array(face_pil.resize((224, 224))) / 255.0
        t_pre_end = time.time()

        # Inference with CNN‑only model
        t_fwd_start = time.time()
        with torch.no_grad():
            logit = MODEL(img_tensor)
            prob = torch.sigmoid(logit).item()
            pred_label = "fake" if prob > 0.5 else "real"
        t_fwd_end = time.time()

        # Grad‑CAM overlay (manual, no external lib)
        t_cam_start = time.time()
        try:
            with torch.enable_grad():
                # Ensure gradients are enabled for the CNN parameters
                for p in MODEL.parameters():
                    p.requires_grad_(True)
                heatmap_b64 = gradcam_overlay(MODEL, img_tensor.clone(), None, orig_np)
                for p in MODEL.parameters():
                    p.requires_grad_(False)
        except Exception:
            heatmap_b64 = None
        t_cam_end = time.time()
        
        t_total_end = time.time()
        
        # Logging
        latency_log = {
            "haar_ms": round((t_haar_end - t_haar_start) * 1000, 2),
            "preprocess_ms": round((t_pre_end - t_pre_start) * 1000, 2),
            "forward_ms": round((t_fwd_end - t_fwd_start) * 1000, 2),
            "gradcam_ms": round((t_cam_end - t_cam_start) * 1000, 2),
            "total_ms": round((t_total_end - t_start) * 1000, 2)
        }
        print(f"[Latency Breakdown] {latency_log}")

        return {
            "label": pred_label,
            "confidence": prob if pred_label == "fake" else 1 - prob,
            "probability_fake": prob,
            "inference_time_ms": latency_log["total_ms"],
            "heatmap_base64": heatmap_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
