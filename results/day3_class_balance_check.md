# Day 3.5: Class Balance Verification

## Extraction Diagnostics
- **Total Processed:** 2000
- **Total Successful Extractions:** 1983 (99.15%)
- **Total Failed (No Face Detected):** 17 (0.85%)

## Class Breakdown (Successful Extractions Only)
- **Fake:** 1983 (100.0%)
- **Real:** 0 (0.0%)
- **Manipulation Type:** N/A (Kaggle Dataset does not track specific subtypes)

## Assessment
The 2,000-image subset is **entirely skewed (100% fake / 0% real)** and is **NOT usable** as-is for training or validation. 

Because `os.walk` iterates through directories sequentially, it traversed deep into the `fake` subdirectories first and hit the 2,000 image limit before it ever saw a single image from the `real` directories. 

## Recommendation
We must re-sample the dataset to ensure a balanced extraction. I recommend modifying `data/batch_extract.py` to first build a full list of all available images across all classes, explicitly separate them into `real` and `fake` lists, shuffle them, and then process a balanced subset (e.g., exactly 1,000 real and 1,000 fake images).
