# Day 11 Notes: Manipulation-Type Breakdown (N/A)

Day 11 was originally scheduled for a "per-manipulation-type accuracy breakdown." 

However, as confirmed in Day 5's leakage investigation and the Day 10 reframing, **this is intentionally skipped because it is not applicable to our dataset.**

Our dataset consists of:
- **REAL**: NVIDIA FFHQ (Genuine photographs)
- **FAKE**: "1 Million Fake Faces" (StyleGAN2 generated synthetic faces)

This is a single generation method (StyleGAN2) from scratch. There are no video-based deepfake manipulation techniques (such as FaceForensics++'s Deepfakes, Face2Face, FaceSwap, or NeuralTextures) present in this dataset. Therefore, a per-manipulation-type breakdown is impossible, and we evaluate the model purely on binary Real vs. Synthetic accuracy.
