# Model Card: AI-Generated Face Detector

## Model Details
- **Architecture:** EfficientNet-B0 (ImageNet pre-trained backbone, fine-tuned end-to-end)
- **Classifier Head:** Linear (1280 → 1) with Sigmoid output.
- **Model Version:** Day 32 Converged (`day32_finetuned_converged.pth`)
- **Input Size:** 224x224 RGB image (cropped to face).
- **Parameters:** ~4.0M
- **Face Extraction Pipeline (Training & Evaluation):** MTCNN
- **Face Extraction Pipeline (Production Deployment):** OpenCV Haar Cascades
- **Maintainer:** Ramsha
- **License:** Open Source (MIT)

## Intended Use
- **Primary Use Case:** Binary classification to detect whether a face is a genuine photograph of a real person (0) or an AI-generated synthetic face (1).
- **Target Domain:** Detecting GAN-generated synthetic faces (specifically StyleGAN2-like spectral upsampling artifacts).
- **Intended Users:** Researchers, platform trust & safety teams exploring synthetic persona risks, and hobbyists.
- **Out-of-Scope Uses:** 
  - ⚠️ **NOT for use as a sole decision-maker in high-stakes real-world moderation.** Given the documented cross-generator failures, this model is blind to newer generative architectures.
  - **NOT for detecting video deepfakes** (e.g., facial reenactment, FaceSwap, FaceForensics++). Video manipulation leaves different artifacts (blending boundaries) than GAN synthesis.

## Training Data
See [DATASET_CARD.md](DATASET_CARD.md) for full details on dataset provenance, licensing, and composition. The model was trained on a balanced 2,000-image dataset (1,000 real faces from FFHQ, 1,000 fake faces from StyleGAN2), split across Train (1,400), Val (300), and Test (300).

## Evaluation Results

### Core Metrics (MTCNN Face Extraction, n=300 balanced)
Results are reported as the **mean ± standard deviation** across 3 independent training runs with different random seeds, confirming high stability.

| Metric | Mean ± Std |
|---|---|
| **Accuracy** | 84.33% ± 0.34% |
| **ROC-AUC** | 0.9321 ± 0.0053 |
| Precision | 0.8140 ± 0.0045 |
| Recall | 0.8378 ± 0.0905 |
| F1 Score | 0.8242 ± 0.0476 |

*(Note: Recall and F1 display higher variance due to their sensitivity to the fixed 0.5 decision threshold. ROC-AUC confirms stable intrinsic model ranking.)*

### Production Performance (Haar Cascade Extraction)
When deployed in a memory-constrained production environment (512MB RAM), MTCNN is replaced with a lightweight Haar Cascade. This induces a fully measured **-6.00 percentage point** accuracy penalty.
- **Production Accuracy:** 78.00%
- **Production ROC-AUC:** 0.8387

### Robustness Evaluation (Retention)
Tested across 9 perturbation conditions relative to a clean baseline:
- **JPEG Compression (q=90 to 30):** Moderate degradation. Accuracy retention 91.6% down to 82.8%.
- **Catastrophic Collapse:** Heavy blur (σ≥2) or extreme downscaling (0.25x) destroys the high-frequency textural signals the model relies on, causing real-image accuracy to collapse to 10-23%.

## Limitations and Ethical Considerations

### 1. Cross-Generator Generalization Failure (Out-of-Distribution Collapse)
The model's capabilities are strictly scoped to faces resembling its StyleGAN2 training distribution. When evaluated on a held-out set of StyleGAN3 generated faces, the model's accuracy collapsed to **46.67%**, and its fake detection accuracy plummeted to just **4.00%** (ROC-AUC 0.5513). It relies heavily on transposed-convolution grid artifacts that StyleGAN3's alias-free architecture suppresses.

### 2. Eyeglasses False-Positive Bias
Grad-CAM interpretability analysis confirmed the model relies on a spurious correlation: it heavily associates eyeglasses (specifically the nose bridge and rims) with synthetic faces. This is caused by a dataset bias where real (FFHQ) images contain many eyeglasses, while StyleGAN2 struggles to render symmetric frames. **Real people wearing glasses will experience elevated false-positive (fake) prediction rates.**

### 3. Ethical Considerations & Fairness Gaps
- **Demographic Bias:** No formal demographic parity or intersectional fairness audit has been performed across gender, skin tone, or age. Relying on this model for automated content moderation poses a risk of disparately impacting certain demographic groups.
- **Dual Use:** Releasing the weights of a discriminator can be used in an adversarial framework to train stronger, more evasive generators.

## Summary
The EfficientNet-B0 model successfully detects StyleGAN2 artifacts with 84.33% accuracy, but relies on brittle, architecture-specific texture signals and dataset biases (eyeglasses) that prevent robust real-world generalization to new generative architectures.
