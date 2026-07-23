# Day 14: Grad-CAM Variant Comparison

We extended our interpretability analysis to compare three distinct Class Activation Mapping variants: **Grad-CAM**, **Grad-CAM++**, and **Eigen-CAM**. The goal was to cross-validate our findings from Day 13 (specifically the spurious "glasses" bias) and determine which variant provides the most precise visual explanations for our EfficientNet-B0 model.

## Observations by Variant

### 1. Grad-CAM (Baseline)
- Consistently produced the most tightly localized and specific heatmaps.
- **Cross-validation**: Perfectly highlighted the glasses frames and nose bridges on both the False Positives (real people with glasses) and True Positives (fakes with glasses). This strongly reaffirms that the model specifically looks at glasses to make its "Fake" prediction.

### 2. Grad-CAM++
- Designed to handle multiple occurrences of an object better, it produced heatmaps that were highly consistent with plain Grad-CAM but slightly more diffuse/broader.
- **Cross-validation**: It fully agreed with the baseline Grad-CAM regarding the glasses bias, heavily activating on the eyeglass rims and surrounding eye regions. The agreement between Grad-CAM and Grad-CAM++ makes the bias finding highly credible.

### 3. Eigen-CAM
- Computes the principal components of the 2D activations without relying on class-specific backpropagation weights. 
- **Result**: It largely failed to explain the model's specific classification decisions. Instead of highlighting the discriminative features (glasses, hairlines, blending edges), Eigen-CAM tended to generically highlight the center of the face (nose/mouth) or high-contrast background edges in almost every image, regardless of the prediction or true label. 

## Conclusion
Both class-discriminative methods (Grad-CAM and Grad-CAM++) completely agree on the presence of the spurious dataset bias (glasses = fake), validating our Day 13 findings. Eigen-CAM was not useful for this specific binary classification explanation task, as it highlighted generic spatial features rather than class-specific signals.

**Decision for Future Visualizations**: We will proceed using plain **Grad-CAM** as our primary interpretability tool. It provides the cleanest, most sharply localized heatmaps for identifying specific synthetic artifacts and spurious correlations without the slight diffusion seen in Grad-CAM++.
