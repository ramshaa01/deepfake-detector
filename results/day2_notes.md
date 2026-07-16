# Day 2 Notes: Face Extraction Pipeline (UPDATED)

**Summary of Processing (Post-Fix):**
- **Total Images Processed:** 30 (Subset of Kaggle dataset `test/fake` split)
- **Genuine Detected Faces (Haar Cascade):** 30
- **Failures (No face detected):** 0
- **Success Rate:** 100%

**Environment Fix & Detector Analysis:**
1. **Environment Correction:**
   - The environment was successfully recreated using Python 3.11. All dependencies from `requirements.txt` (including `tensorflow` and `mtcnn`) successfully installed.

2. **MTCNN Application Control Block:**
   - Despite being successfully installed in the Python 3.11 environment, MTCNN still failed to run. The underlying `tensorflow` dependency crashed with an OS-level error: `ImportError: DLL load failed while importing _pywrap_parallel_device: An Application Control policy has blocked this file.`
   - This indicates a strict Windows Defender or AppLocker policy on the host machine blocking the execution of certain TensorFlow DLLs.

3. **Haar Cascade Success & Fallback Removal:**
   - As requested, the silent "resize the whole image" fallback was completely removed from the pipeline. 
   - The script now relies on MTCNN (primary) and OpenCV Haar Cascades (secondary).
   - Because MTCNN crashed due to the OS policy block, the script fell back to the Haar Cascades.
   - Surprisingly, the Haar Cascades successfully identified a face bounding box in all 30/30 sample images this time! The newly saved sample crops in `results/day2_sample_crops/` reflect genuine detections, not silent resize fallbacks.

The pipeline is now honest: it either detects a face, or fails and skips the image.
