# Day 3 Notes: Batch Face Extraction & Metadata Tracking (CORRECTED)

## Dataset Processing Details
- **Script:** `data/batch_extract.py`
- **Extraction Tools:** MTCNN (Primary — now working via TensorFlow 2.15 on Python 3.11)
- **Strategy:** Balanced random sampling — 1,000 real + 1,000 fake images sampled with `random.seed(42)` for reproducibility
- **Run Time:** ~38 minutes for 2,000 images (MTCNN CPU inference, ~1.15s/image)

## True Dataset Population (Before Sampling)
The full walk of `data/raw_samples/` found all images across train/test/valid splits. Sampling was done across all splits to avoid train/test leakage concerns — the downstream PyTorch DataLoader (Day 4) will implement the formal stratified split.

## Extraction Metrics (Balanced 2,000 image batch)

| Metric | Count | Percentage |
|---|---|---|
| Total Processed | 2000 | 100% |
| Total Successful | 2000 | 100.00% |
| Total Failed | 0 | 0.00% |

## Class Balance (Final)

| Label | Successful | Failed | % of Total Success |
|---|---|---|---|
| REAL | 1000 | 0 | 50.00% |
| FAKE | 1000 | 0 | 50.00% |

- **Class Ratio:** real: 1000 (50%), fake: 1000 (50%) ✅ Perfectly balanced

## Key Findings
- MTCNN (the primary deep learning detector) is now fully operational on this machine via TensorFlow 2.15 on Python 3.11.
- 0/2000 failures — MTCNN achieved 100% face detection success on the balanced sample, significantly outperforming the Haar Cascade secondary detector.
- The dataset is ready for stratified train/val/test split and PyTorch DataLoader construction (Day 4).

## Previous Run (DISCARDED — unbalanced)
The first run (2,000 images via raw `os.walk` order) was 100% fake due to directory traversal ordering hitting the fake folder first. That run was replaced entirely by this balanced re-extraction.
