# Day 13: Grad-CAM Interpretability Review

We integrated `pytorch-grad-cam` to visualize what the fine-tuned EfficientNet-B0 model is "looking at" when making its predictions. We used the final convolutional block (`model.conv_head`) as the target layer.

## Observations from Grad-CAM Overlays

### 1. Correctly Classified Samples
- **Correct Fakes**: For StyleGAN2 fakes that the model correctly identified, the heatmaps are highly concentrated on areas known to be difficult for GANs. For example, in one image of a woman with glasses, the heatmap intensely highlights the **glasses and frames**. In another fake, it focuses heavily on the **hairline and forehead boundaries**, which are classic regions for GAN blending artifacts.
- **Correct Reals**: The model focuses on high-frequency detail areas like the mouth/teeth or the eyes and ears, likely picking up on natural, asymmetric, or highly detailed textures that GANs often smooth out.

### 2. Hardest Errors (From Day 12)
- **False Positives (Real flagged as Fake)**: As suspected in Day 12, the model has learned a strong (but flawed) heuristic: **Glasses = Fake**. In the hardest false positives (older men with glasses), the Grad-CAM heatmaps light up like a Christmas tree directly over the nose bridge and the rims of the glasses. Because StyleGAN struggles to generate symmetrical glasses, the model learned that glasses are highly correlated with fakes, and is now unfairly penalizing real people wearing them.
- **False Negatives (Fakes flagged as Real)**: 
  - *Harsh Shadows*: For the fake image with heavy directional lighting/shadows, the heatmap completely ignores the face and focuses on the brightly lit background edge. The model was distracted or couldn't find its usual facial cues in the shadows.
  - *Babies/Toddlers*: For the fake toddler, the heatmap focuses perfectly on the mouth and cheeks, but finds nothing "wrong." Since StyleGAN excels at rendering smooth baby skin, the model saw a perfectly rendered mouth and incorrectly assumed it was a real photograph.

## Conclusion
The Grad-CAM heatmaps are highly interpretable and successfully explain *why* the model is making errors. The model isn't failing randomly; it has learned specific, logical heuristics (e.g., examining glasses, hairlines, and skin texture) that sometimes backfire on edge cases. 

This confirms that the model is functioning as a genuine feature-extractor for synthetic artifacts, rather than just memorizing the training set.
