# Inference API

This directory contains the FastAPI serving layer for the AI-Generated Face Detector.

## Live Deployment

The API is live on Render:

**Base URL:** `https://deepfake-detector-k62g.onrender.com`

> ⚠️ **Cold-Start Warning:** Render's free tier spins the service down after 15 minutes of
> inactivity. The **first request after idle time will take 30–90 seconds** to respond
> while the container wakes up. Subsequent requests within the session are fast (~200–400ms).
> Always hit `/health` first when demoing.

## Running Locally

To start the server locally (requires all dependencies from `requirements.txt`):
```bash
# From the project root
.\\venv\\Scripts\\activate
uvicorn inference.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note (local only):** The local server uses MTCNN (TensorFlow-backed) as the primary face
> detector with Haar Cascade fallback. The deployed Render version uses Haar Cascade only
> (MTCNN/TensorFlow removed to stay within Render's 512 MB free-tier RAM limit).
> Prediction outputs are numerically identical — only face detection recall differs slightly.

## Endpoints

### 1. `GET /health`
Returns the basic uptime status and loaded model information.

**Example Request:**
```bash
curl https://deepfake-detector-k62g.onrender.com/health
# or locally:
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
Analyzes an uploaded face image and returns a prediction along with a Grad-CAM heatmap.

**Rate Limit:** 10 requests per minute per IP address. Exceeding this limit will return a `429 Too Many Requests` response.

**Example Request:**
```bash
curl -X POST "https://deepfake-detector-k62g.onrender.com/predict" \
     -H "accept: application/json" \
     -F "file=@/path/to/face.jpg;type=image/jpeg"
```

**Example Response:**
```json
{
  "label": "real",
  "confidence": 0.7738,
  "probability_fake": 0.2262,
  "inference_time_ms": 384.92,
  "heatmap_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

**Constraints:**
- Maximum file size: 10 MB
- Accepted types: `image/jpeg`, `image/png`, `image/webp`, etc. (any `image/*` MIME type)
- The uploaded image must contain a clearly visible face. If no face is detected, a
  `400 Bad Request` is returned.

## Error Codes
- `400 Bad Request`: If the file is not an image, or if no face could be detected by the Haar Cascade.
- `413 Payload Too Large`: If the uploaded file exceeds the 10MB limit.
- `429 Too Many Requests`: If the client exceeds the rate limit (10 requests/minute).
- `500 Internal Server Error`: For any unexpected backend errors.

## Response Fields

| Field | Type | Description |
|---|---|---|
| `label` | `"real"` or `"fake"` | Model prediction |
| `confidence` | float (0–1) | Confidence in the predicted label |
| `probability_fake` | float (0–1) | Raw probability that the face is synthetic |
| `inference_time_ms` | float | Time spent in model inference (excludes face detection) |
| `heatmap_base64` | string | Base64-encoded JPEG of Grad-CAM activation overlay |

## Known Limitations

This endpoint uses the `day16_fusion_best.pth` model. Documented limitations:
- **Eyeglasses Bias:** The model has a learned shortcut associating eyeglasses with synthetic
  faces (dataset distribution mismatch between FFHQ and StyleGAN2).
- **Blur/Downscale Collapse:** Under heavy Gaussian blur (σ≥2) or severe downscaling (0.25×),
  the model collapses and predicts nearly everything as `fake`. This is a fundamental limitation
  of the spectral-artifact features the model learned.

For full robustness metrics, see [results/day19_robustness_metrics.json](../results/day19_robustness_metrics.json).
