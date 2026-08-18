# Day 39: Inference Latency Breakdown

To demystify the ~256ms single-image inference latency observed in production, we benchmarked the internal components of the FastAPI endpoint over 10 consecutive requests (CPU).

## Component-Wise Latency (Average of 10 runs)

| Pipeline Stage | Average Latency (ms) | Description |
|---|---|---|
| **1. Haar Cascade Face Detection** | 61.47 ms | OpenCV Haar cascade to detect bounding box + crop + resize. |
| **2. Image Preprocessing** | 2.35 ms | PyTorch `transforms` (ToTensor, ImageNet Normalize). |
| **3. EfficientNet-B0 Forward Pass** | 80.87 ms | Model inference (`torch.no_grad`). |
| **4. Grad-CAM Overlay Generation** | 300.90 ms | Secondary pass `torch.enable_grad()`, backwards hook, heatmap blending, base64 encoding. |
| **Total API Internal Latency** | **445.59 ms** | Total internal time (excludes network/FastAPI overhead). |

### Interpretation
The total internal latency matches our expectations. Grad-CAM generation is the most expensive single operation because it requires re-enabling gradients, performing a backwards pass on the CNN to compute attribution weights, then performing NumPy/OpenCV blending and JPEG base64 encoding. The forward pass itself is relatively lightweight (~75ms). 
