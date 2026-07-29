# Day 21: Full Pipeline Consistency Audit

Before moving onto the FastAPI and frontend serving phase, a complete consistency audit of the ML pipeline (Days 1–20) was performed.

## 1. Preprocessing Consistency
**Status:** ✅ Fully Consistent
- **Resize:** All scripts that ingest images (`dataset.py` for training/eval, `evaluate.py`, `evaluate_fusion.py`, `run_robustness_eval.py`, `gradcam_viz.py`) correctly resize images to `224x224`.
- **Normalization:** All evaluation scripts use `get_val_test_transforms()` from `dataset.py`, which applies consistent ImageNet mean/std normalization.
- **FFT Extraction:** Both the Day 15 baseline `frequency_features.py` and the Day 19 `run_robustness_eval.py` (via `day20_analysis.py`) use the exact same `extract_radial_profile()` logic (resizing to 256x256 before FFT, extracting 16 radial bins).

## 2. Label Convention
**Status:** ✅ Fully Consistent
- `dataset.py` defines `LABEL_MAP = {"real": 0, "fake": 1}`. All `DataLoader` instances across the pipeline use this mapping. No inversion of precision/recall was found.

## 3. Dataset Split Integrity
**Status:** ✅ Fully Consistent
- The git history for `data/splits/train.csv`, `val.csv`, and `test.csv` confirms they have not been touched or overwritten since they were generated on Day 4. The 300 test images used on Day 7, Day 10, Day 16, and Day 19-20 are identical.

## 4. End-to-End Predictability (Spot-Check)
**Status:** ✅ Consistent (with one typo flagged in README)
- Re-running `evaluate_fusion.py` fresh produced the exact same aggregate results as recorded on Day 16:
  - **Accuracy:** 79.33%
  - **ROC-AUC:** 0.8544
  - **Precision:** 0.7558
  - **Recall:** 0.8667
- **Note on README Typo:** The README previously recorded the Day 16 test accuracy as 79.00%. This 79.00% was actually the `val_acc` of the best checkpoint (`epoch 12 (val_acc=0.7900)`). The true test accuracy has always consistently been 79.33%. The pipeline is solid; only the markdown summary had a transcription error.

## Conclusion
The core ML pipeline is fully deterministic, consistent, and ready to be wrapped in a production API. No regressions or silent drifts have occurred.
