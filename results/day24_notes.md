# Day 24 Deployment Notes — FastAPI Backend on Render

## Summary

Day 24 successfully deployed the FastAPI inference endpoint (built on Day 22) to
a live cloud host. After a pivot from HuggingFace Spaces, the backend is running on
Render's free tier at:

**Live URL:** https://deepfake-detector-k62g.onrender.com

---

## Deployment Approach

### Platform Selection: Pivot from HuggingFace Spaces to Render

The original plan was to deploy to HuggingFace Spaces using the Docker SDK. This failed
immediately:

```
Client error '402 Payment Required'
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
cpu-basic requires a PRO subscription.
```

HuggingFace silently changed their billing policy — Docker Spaces now require a PRO
subscription even on the free hardware tier. We pivoted to **Render's free Web Service**
tier, which still accepts Docker containers and GitHub repo auto-deploy at no cost.

### Dockerfile Structure

The final `Dockerfile` (project root) uses Python 3.11-slim and installs dependencies
in a deliberate order to avoid the `libGL.so.1` error caused by `grad-cam`'s transitive
`opencv-python` dependency overriding our headless install:

```dockerfile
# 1. Install system libs (libgl1, libglib2.0-0) to satisfy any OpenCV C-level deps
RUN apt-get install -y libgl1 libglib2.0-0

# 2. Install CPU-only PyTorch (avoids ~1.5GB CUDA wheels)
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Install headless OpenCV BEFORE grad-cam (critical ordering)
RUN pip install opencv-python-headless==4.10.0.84

# 4. Install everything else (grad-cam sees headless already installed)
RUN pip install -r requirements-deploy.txt
```

A separate `requirements-deploy.txt` was created (excludes torch, torchvision, opencv
since those are handled in the Dockerfile with special ordering).

### Model Checkpoint Strategy

`day16_fusion_best.pth` (~16 MB) and `data/frequency_features.npz` (~265 KB) were
**force-committed directly to the GitHub repo** using `git add -f`. This is simpler
than setting up HuggingFace Hub download-at-startup logic, and both files are
well under GitHub's 100 MB per-file limit. The general `.gitignore` rule for `*.pth`
files still applies to all other checkpoints.

### Memory Optimization for 512 MB Free Tier

Render's free tier OOM-killed our container (exit status 137) when TensorFlow+MTCNN
were loaded alongside PyTorch. The fix:

- **Removed MTCNN/TensorFlow entirely** from the deployed inference path. Face
  extraction now uses only OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
- **Replaced `pytorch-grad-cam` library** with a manual Grad-CAM implementation using
  PyTorch forward/backward hooks (~50 lines, zero extra dependencies, equivalent output).
- Set `torch.set_num_threads(1)` and `ENV MALLOC_ARENA_MAX=2` to reduce glibc
  memory fragmentation overhead.
- CPU-only PyTorch wheels (~800 MB smaller than the default CUDA build).

Peak RAM usage after these changes: ~350–380 MB (safely within 512 MB limit).

---

## Build Issues Encountered & Resolved

| # | Error | Root Cause | Fix |
|---|---|---|---|
| 1 | `ImportError: libGL.so.1` | `grad-cam` pulls in full `opencv-python`, which overwrites headless install | Install headless BEFORE grad-cam in Dockerfile; add `libgl1` system package |
| 2 | Exit status 137 (OOM) | TensorFlow+MTCNN+PyTorch combined RAM usage exceeded 512 MB | Remove TF/MTCNN; use Haar Cascade only for face detection on Render |
| 3 | Port not binding | Shell variable not expanded in JSON-form CMD | Changed CMD to `sh -c "uvicorn ... --port ${PORT}"` |
| 4 | Heatmap returned empty | Grad-CAM was calling `model.cnn(img_tensor)` which returns 1280-dim flat vector, not a spatial map with valid gradients | Fixed to call full `model(img_tensor, freq_tensor)` for correct scalar logit and proper gradient flow through `conv_head` |
| 5 | HuggingFace 402 | HF Spaces Docker SDK now requires PRO subscription | Pivoted to Render |

---

## Live Inference Verification

Tests run against `https://deepfake-detector-k62g.onrender.com` using the same images
from Day 22's local validation:

### `/health`
```json
{"status": "ok", "model": "day16_fusion_best.pth", "device": "cpu"}
```

### `/predict` — REAL image (`data/faces_extracted/real/00022.jpg`)
```
Local (Day 22):  label=real  confidence=0.7843  prob_fake=0.2157
Live  (Day 24):  label=real  confidence=0.7738  prob_fake=0.2262
```
> Minor confidence difference is expected: local used MTCNN face crop, Render uses Haar
> Cascade crop. The face region differs slightly, causing a small shift in confidence,
> but the **label is identical (real)**.

### `/predict` — FAKE image (`data/faces_extracted/fake/01HYNP6M67.jpg`)
```
Local (Day 22):  label=fake  confidence=0.9664  prob_fake=0.9664
Live  (Day 24):  label=fake  confidence=0.9708  prob_fake=0.9708
```
> Again, minor confidence shift due to Haar vs MTCNN crop. **Label matches (fake)**.

### `/predict` — Blank image (no face)
```
Local (Day 22):  400 Bad Request — "No face detected in the image or failed to process."
Live  (Day 24):  400 Bad Request — "No face detected in the uploaded image. Please upload an image containing a clear, visible face."
```
> Behaviour is consistent: both reject non-face images without silently falling back to a centre-crop.

Both real/fake predictions are **correct and directionally consistent** with local results.

---

## Cold-Start Caveat

Render's free tier automatically **spins down the container after 15 minutes of
inactivity**. The first request after idle time triggers a cold start, which takes
**30–90 seconds** depending on model load time. This is documented in both
`inference/README.md` and the main `README.md`. Always hit `/health` first during demos.

---

## Commit History

| Commit | Description |
|---|---|
| `84594c1` | Initial Dockerfile + checkpoint push |
| `d9fdc44` | Added libgl1 system dep (partial fix) |
| `ffb2474` | Removed TF/MTCNN, limited torch threads (OOM fix attempt) |
| `a5b696b` | Added libgl1 + libglib2.0 + MALLOC_ARENA_MAX |
| `6a6a264` | Final fix: correct pip install ordering + manual Grad-CAM |
