# Inference API

This directory contains the FastAPI serving layer for the AI-Generated Face Detector.

## Running Locally

To start the server locally:
```bash
# From the project root
.\venv\Scripts\activate
pip install fastapi uvicorn python-multipart
uvicorn inference.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

### 1. `GET /health`
Returns the basic uptime status and loaded model information.

**Example Request:**
```bash
curl http://localhost:8000/health
```

**Example Response:**
```json
{
  "status": "ok",
  "model": "day16_fusion_best.pth",
  "device": "cpu"
}
```

### 2. `POST /predict`
Accepts an image file upload, extracts the face (using MTCNN/Haar cascades), runs it through the fusion model, and returns the prediction and a base64-encoded Grad-CAM heatmap overlay.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/faces_extracted/real/00000.png;type=image/png"
```

**Example Response:**
```json
{
  "label": "real",
  "confidence": 0.98,
  "probability_fake": 0.02,
  "inference_time_ms": 250.45,
  "heatmap_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

## Known Limitations

This endpoint relies on the `day16_fusion_best.pth` model. Please be aware of the documented limitations:
- **Eyeglasses Bias:** The model has a learned shortcut to associate eyeglasses with synthetic faces (due to a dataset distribution mismatch between FFHQ and StyleGAN2).
- **High-Frequency Degradation Collapse:** The model collapses to a catastrophic false-positive state (predicting everything as fake) under heavy Gaussian blur or severe downscaling (e.g. `0.25x`).

For full details on the robustness metrics and evaluation limitations, see the [Main README](../README.md#known-limitations).
