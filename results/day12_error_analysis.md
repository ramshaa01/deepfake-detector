# Day 12: Error Analysis on Fine-Tuned Model

## Error Rates

We evaluated the fine-tuned checkpoint (`day8_finetuned_best.pth`) on the isolated test set of 300 images (150 Real, 150 Fake). 

- **Total Test Samples**: 300 (150 Real, 150 Fake)
- **False Positives** (Real flagged as Fake): 33 -> **FPR: 22.00%**
- **False Negatives** (Fake flagged as Real): 33 -> **FNR: 22.00%**

*Note: The model is perfectly balanced in its error distribution, showing no systemic bias toward over-flagging or under-flagging.*

## Visual Inspection of Hardest Errors

We extracted the top 6 most confident False Positives and False Negatives (plotted in `results/day12_hardest_errors.png`) to identify any visual patterns in where the model fails most confidently.

### False Positives (Real images confidently flagged as Fake)
- **Glasses & Reflections**: Three of the most confident FPs are of individuals wearing glasses. StyleGAN is known to struggle with rendering symmetrical, realistic glasses frames and reflections, which may cause the model to over-index on glasses as a potential "fake" signal.
- **Skin Texture / Makeup**: One image features a woman with heavy makeup and extremely smooth skin. The model likely confused this natural (or cosmetically enhanced) smoothness with the "plastic" or overly-smoothed texture typical of generated faces.
- **Age/Wrinkles**: Several older individuals with pronounced wrinkles were flagged.

### False Negatives (StyleGAN Fakes confidently flagged as Real)
- **Extreme Lighting/Shadows**: Several FNs feature harsh, directional lighting or heavy shadows on one side of the face. StyleGAN can sometimes render highly convincing lighting, and the model might be associating these complex lighting environments strictly with real photography.
- **Children/Babies**: Two of the most confident FNs are of a toddler and a baby. StyleGAN often renders children very well (smooth skin naturally), which seems to fool the detector easily.
- **Obvious Artifacts Missed**: Surprisingly, one image with severe, classic StyleGAN artifacts (purple/yellow discoloration and a highly distorted mouth) was confidently predicted as Real. This suggests the model may not have learned to detect these specific gross distortions, possibly because it's relying on more subtle frequency or texture cues instead.

## Conclusion
While the overall accuracy is solid (78%), the error analysis reveals that the model struggles with specific demographic features (glasses, babies) and lighting conditions. Grad-CAM analysis on Day 13 will be critical to see exactly *where* the model is looking in these failure cases (e.g., is it actually looking at the glasses in the FPs?).
