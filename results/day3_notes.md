# Day 3 Notes: Batch Face Extraction & Metadata Tracking

## Dataset Processing Details
- **Method:** Iterated over the Kaggle Deepfake dataset subset using `data/batch_extract.py`.
- **Extraction Tools:** MTCNN (Primary - still blocked by OS policy), OpenCV Haar Cascades (Secondary - Active).
- **Execution Run:** To establish baseline metrics and a usable labeled batch for Day 3, a subset sample of 2,000 images was processed. The script supports processing the entire 140k dataset via removing the `--limit` flag (note: full run takes ~2-3 hours).

## Extraction Metrics (2,000 image batch)
- **Total Images Processed:** 2000 (all from the `fake` split due to directory traversal order)
- **Total Successfully Extracted:** 1983
- **Total Failed (No Face Detected):** 17
- **Success Rate:** 99.15%
- **Failure Rate:** 0.85%

## Failure Analysis
- The failure rate is **less than 1%**, which is phenomenally good for OpenCV Haar Cascades on deepfake datasets.
- Because the failure rate is well below the 15-20% threshold, no extensive manual failure investigation was necessary. The few failures (17 images) are likely due to extreme angles, occlusions, or low lighting where Haar Cascades typically struggle.

## Metadata Tracking
- Output images were saved into `data/faces_extracted/fake/`.
- A metadata CSV (`data/faces_extracted/metadata.csv`) was successfully generated, tracking filename, label, source path, and extraction status for all 2000 processed images. This file provides clean, structured labels for PyTorch Dataset/DataLoader creation in Day 4.
