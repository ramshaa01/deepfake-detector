# Interview Story Bank — AI-Generated Face Detector

> **For internal prep use only.** All numbers and events are grounded in the actual
> git history of [github.com/ramshaa01/deepfake-detector](https://github.com/ramshaa01/deepfake-detector).

---

## Numbers to Have Memorized

| Item | Value |
|---|---|
| **Production model** | `day32_finetuned_converged.pth` (EfficientNet-B0, CNN-only) |
| **Official test accuracy** | 84.00% (MTCNN face detector, 300-image balanced held-out set) |
| **Official ROC-AUC** | 0.9372 |
| **Production accuracy** | 78.00% (Haar Cascade face detector, same test set) |
| **Production ROC-AUC** | 0.8387 |
| **Why the gap exists** | Render's 512MB free-tier RAM cannot fit MTCNN/TensorFlow; switched to OpenCV Haar Cascade, which fails to detect 4.7% of faces and crops differently on the rest — fully measured, not estimated |
| **Why fusion was rejected** | CNN-only wins ROC-AUC (0.9372 vs 0.9328), wins precision at every matched-recall point, and is 17% faster — fusion's higher default-threshold accuracy was a threshold artifact, not a genuine ranking gain |

---

## STAR Stories

---

### Story A — The Class-Skew Bug (Day 3)

**Situation.**
On Day 3, I had just finished writing the data pipeline: an `os.walk` over the raw
dataset directory to collect file paths, then a stratified split into train/val/test
CSVs. The dataset was supposed to be balanced — 1,000 real faces from FFHQ and 1,000
GAN-generated faces from StyleGAN2. I was about to move straight into training.

**Task.**
Before writing a single training loop, I wanted to verify the splits were actually
balanced. A silent class imbalance at this stage would corrupt every downstream metric
invisibly — the model could hit 80% accuracy by predicting everything as "fake" and I'd
never know from the loss curve alone.

**Action.**
I wrote a two-line diagnostic: iterate each CSV, count labels, print the real-to-fake
ratio per split. The output was immediate and alarming: train split showed 0% real,
100% fake. The `os.walk` was traversing subdirectories in alphabetical order and my
label assignment logic was keying off directory depth rather than the folder name — the
"real" subdirectory happened to sort after "fake" and the slicing logic cut it off
entirely. I fixed the label extraction to key explicitly off the directory name string,
re-ran the diagnostic, confirmed 50/50 balance across all three splits, and only then
wrote the first training script.

**Result.**
Zero downstream impact because the bug was caught pre-training. If it had slipped
through, every metric from Day 6 onward — accuracy, AUC, confusion matrix — would have
been meaningless. The diagnostic took 10 minutes to write and saved potentially weeks of
debugging why a "trained" model had degenerate behavior. I kept the diagnostic as a
permanent assertion in the data pipeline so any future re-split triggers the same check
automatically.

---

### Story B — The Inference-Timing and Threshold-Artifact Investigation (Day 16-17)

**Situation.**
After training the CNN+FFT fusion model on Day 16, I reported 79.33% accuracy and noted
an inference time of ~53 ms. Both numbers looked good. But when I set up the FastAPI
endpoint on Day 22 and benchmarked it under realistic conditions — single image, no
pre-computed FFT, loading from disk — the measured latency was 241 ms, nearly five
times higher. At the same time, the fusion model's recall (86.67%) was substantially
higher than the CNN-only baseline (78.00%), which I'd initially reported as a clear win
for fusion.

**Task.**
I needed to determine: was the recall improvement real, or was it a threshold artifact?
And was the latency discrepancy a measurement error on my part, or something more
fundamental?

**Action.**
On the latency side, I traced every step of the production path: the 53 ms figure came
from batch evaluation with batch-size 32 and pre-computed FFT features loaded from a
`.npz` file — conditions that don't exist in the API. The correct apples-to-apples
benchmark was batch-size 1, on-the-fly FFT extraction. I re-ran under those conditions:
241 ms. I documented both figures with methodology in `results/day16-17_timing_investigation.md`
and updated the README to clearly label which context each number came from.

On the recall side, I built a matched-operating-point table: I tuned the CNN-only
decision threshold to exactly match the fusion model's 86.67% recall, then compared
precision at that operating point. CNN-only achieved 73.03% precision; fusion achieved
75.58% — a real but narrow +2.55pp gap. So the recall headline was a threshold artifact,
but a genuine precision advantage did exist at that recall level, which justified the
fusion model's selection at the time.

**Result.**
Two separate findings, each resolved with different tools. The timing bug was a
methodology error — fixed by standardizing benchmarking conditions. The recall
advantage was partially a threshold artifact but partially real — the matched-OP table
gave a defensible, non-cherry-picked basis for the decision. Both findings were
documented, not buried. When I revisited this exact framework on Day 32, it was the
same matched-OP methodology that ultimately overturned the fusion decision.

---

### Story C — The Grad-CAM Glasses-Bias Discovery (Day 13-14)

**Situation.**
By Day 13, the CNN model had reached ~75% validation accuracy, which seemed reasonable.
I ran Grad-CAM to visualize what the model was attending to when it predicted "fake."
The first heatmap that came back was on a real face of a person wearing glasses — and
the activation was almost entirely on the nose bridge and lens rims, not on facial
structure.

**Task.**
Before reporting "the model has a glasses bias" to anyone, I needed to validate that
this wasn't an artifact of the specific Grad-CAM implementation, the specific image, or
random noise in a single sample.

**Action.**
I ran three checks. First, I generated heatmaps across 30 images with glasses and 30
without, and compared the spatial distribution of activations. Second, I implemented
two alternative attribution methods — plain gradient-times-input and occlusion
sensitivity — and checked whether they agreed with the Grad-CAM heatmap on the glasses
region. Third, I looked at the error analysis from Day 11-12: among the hardest false
positives (real faces incorrectly predicted as fake), glasses appeared at a
disproportionately high rate compared to the overall dataset.

All three agreed: the model was using glasses as a spurious proxy for "fake." The root
cause was a distributional mismatch — FFHQ (real) contains far more glasses than
StyleGAN2-generated faces, because StyleGAN2 struggles to synthesize symmetric eyeglass
frames, making glasses an inadvertent but strong signal for "real human." The model
learned the shortcut in reverse.

**Result.**
The finding went into the documented Known Limitations section of the README — not as a
future aspiration but as a specific, characterized failure mode with a root cause. Any
reviewer deploying this model now knows that real people wearing glasses will see
elevated false-positive rates. The cross-validation discipline also paid off later:
when I had a surprising finding on Day 32, my instinct was the same — don't trust a
single metric from a single angle.

---

### Story D — The Fusion-vs-CNN Decision on Day 32 (Using the Same Rigor from Story B)

**Situation.**
The production model since Day 16 had been the CNN+FFT fusion architecture. By Day 31,
I discovered that the Day 8 fine-tuning run had been interrupted by an infrastructure
restart at epoch 7, before the model converged. This meant the fusion model's "84%"
claim was built on a CNN backbone that was never fully trained. I re-ran CNN
fine-tuning to true convergence — 30 epochs, early-stopping patience of 4 — and got
84.00% test accuracy, up from 78.00%. Then I retrained the fusion head on top of this
converged backbone.

**Task.**
With a better backbone, did fusion still help? This was the right question to ask before
changing production, and I was not going to answer it with default-threshold accuracy
alone — I'd already learned from Story B that threshold choice can manufacture apparent
gains.

**Action.**
I ran the same matched-operating-point analysis from Day 16-17: held recall fixed at
three levels (0.800, 0.850, 0.867) and compared precision between CNN-only and fusion
at each point. I also compared ROC-AUC directly, since it is threshold-independent by
construction. The results were the opposite of what I expected: CNN-only had higher
ROC-AUC (0.9372 vs 0.9328) — meaning it ranked real vs fake better at every possible
threshold. CNN-only had higher precision at all three matched-recall points. And CNN-only
was 17% faster (73.5 ms vs 88.05 ms at batch=1). The fusion model's higher default-
threshold accuracy (82% vs 84% in favor of CNN-only) was — as before — a threshold
artifact.

**Result.**
I reversed the deployment decision. The simpler model was strictly better on the
metrics that can't be gamed by threshold tuning, and the more complex architecture added
latency and fragility for no genuine gain. The decision went into the README's model
selection section with the full evidence table, not just the conclusion. The lesson is
that "more architecture" is not automatically better — you have to measure at the right
level of abstraction, and a matched-operating-point analysis is that level for binary
classification problems.

---

## Three Honest "What Would You Improve" Answers

1. **No published baseline comparison.** I never compared against XceptionNet, which is
   the standard academic benchmark for GAN-face detection. My 84% number is only
   meaningful relative to my own earlier checkpoints — I can't tell you where it sits
   on the published literature leaderboard.

2. **No cross-generator generalization test.** Every image this model has ever seen at
   train, val, or test time came from either FFHQ (real) or StyleGAN2 (fake). I have no
   idea how it performs on StyleGAN3, diffusion-model-generated faces, or any other
   generator. For a real deployment, that generalization gap is the most important
   unknown.

3. **The production accuracy gap is an engineering problem, not an inherent limit.**
   Switching from MTCNN to Haar Cascade costs 6 percentage points of accuracy (84%
   to 78%). That gap exists because of a 512MB RAM constraint on the free tier. With a
   paid deployment tier — or by distilling MTCNN into a smaller model — that gap is
   recoverable. It's a resource constraint I've measured precisely; it's not a modeling
   ceiling.

---

## Elevator Pitch (30 seconds)

I built a full-stack deepfake face detector end-to-end — data pipeline, model training,
a FastAPI backend, and a React frontend, all deployed and live. What I'm most proud of
isn't the accuracy number; it's the engineering discipline. I caught a class-skew bug
before it corrupted training, I used a matched-operating-point analysis to separate a
real precision gain from a threshold artifact, I validated a Grad-CAM finding across
three attribution methods before trusting it, and I used that same rigor on Day 32 to
discover that a more complex architecture I'd built actually didn't generalize on a
properly-converged backbone — and I chose the simpler model because the evidence said
to. The final model hits 84% accuracy on a balanced held-out test set, and the
production delta from the memory-constrained deployment is fully measured and disclosed.
