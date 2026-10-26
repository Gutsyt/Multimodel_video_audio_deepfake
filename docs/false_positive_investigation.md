# Why was a real video flagged? Investigation notes (26 Oct 2026)

**Observed (by the project owner, one video):** audio raw score 1.00, visual raw score 0.76, combined 0.88, default threshold 0.50, decision "flagged". The video is described as real.

**What this document can and cannot say.** One video is one data point. There is no labelled evaluation set in this repository, so **nothing here proves that the flag is a false positive, and nothing here measures accuracy.** The sections below separate (1) things that were verified, (2) observations from running the real models, (3) hypotheses that remain unproven, and (4) what would be needed to settle the question.

## 1. Verified: the score mappings are not inverted

| Branch | What the code reports | How it was checked |
|---|---|---|
| Audio (AASIST) | `softmax(logits)[:, 0]` = mass on the **spoof** class. | Upstream `clovaai/aasist` at the vendored commit `a04c986`: `main.py` evaluates `batch_out[:, 1]` as the **bona fide** score and `data_utils.py` labels `bonafide = 1`, `spoof = 0`. Index 0 is therefore spoof, which is what the wrapper uses. Re-read from the upstream files on 26 Oct 2026. |
| Visual (Hugging Face ViT) | `softmax(logits)[:, fake_idx]` where `fake_idx` is the single label containing "fake". | The checkpoint's `config.json` has `id2label = {0: "Realism", 1: "Deepfake"}`; the loader picks index 1 and refuses to run if the label is ambiguous. A real run confirmed `{0: 'Realism', 1: 'Deepfake'}, fake_index=1`. |

Both reported scores are plain softmax outputs. They are **not probabilities of being fake**: neither model was calibrated, and neither was validated on this project's data.

Preprocessing was checked against the upstream evaluation code: 16 kHz mono, 64,600-sample (about 4.04 s) windows, short tail tiled to full length (upstream `pad()` does the same), no amplitude normalisation (upstream does none).

## 2. Observed with the real models (single samples, not evaluations)

- **A genuine speech recording was flagged by the audio branch.** A 13 s historic speech recording (public sample `mlk.flac`; I have no reason to believe it is synthetic) gave per-window raw scores **0.007, 0.993, 0.992**, mean **0.664**, so it was "flagged" at 0.5. The three windows of one continuous recording disagree almost completely, and two are saturated (logit margin about 4.5 to 5). That is the pattern expected of a model operating outside its training domain, but it is still a single recording.
- **The mean hides window disagreement.** The same clip scored 0.664 as served, 0.745 peak-normalised, 0.754 with the window grid shifted by 2 s, and 0.739 with the other checkpoint (AASIST-L). The score moves by up to 0.09 with harmless preprocessing changes, and individual windows move far more.
- **The visual model scored a real photograph above the threshold.** The repository's `face.jpg` scored 0.615 ("Deepfake" logit 0.32 vs "Realism" logit -0.15). The margin is small, so it is a weak preference, but it is on the flagged side of 0.5. One image.
- The audio model's own model card already records that pure noise scores as spoof and cites near-chance equal error rates on in-the-wild audio.

## 3. Plausible contributors (hypotheses, not findings)

1. **Domain shift (audio).** AASIST was trained on clean English ASVspoof 2019 LA speech. Phone/camera audio, compression, background sound and music are outside that domain.
2. **Saturation.** A raw score of 1.00 means one logit dominates; it says the model is confident, not that it is right.
3. **Mean aggregation (audio and visual).** One or two extreme windows/frames can dominate the mean (see the observation above). Median, min, max and range are now reported next to the mean.
4. **Weak face detector (visual).** OpenCV Haar is frontal-only and can return non-face boxes; the largest box is scored. Per-frame box size, crop size and detection counts are now reported so wrong crops can be spotted.
5. **Image-level classifier on video frames (visual).** The model has no temporal reasoning, its training data is undisclosed, and its own card warns about poor generalisation and compressed/low-resolution input.
6. **Equal-weight fusion of two unrelated, uncalibrated models.** With both scores high the fused score is high by construction; the weights and threshold are untrained defaults.

## 4. What was added so the cause can be inspected

- Every audio window and visual frame now carries **both logits and both class probabilities**, plus the verified output-semantics statement.
- Audio: peak level, clipping ratio, loudness, active-frame ratio, a (heuristic, non-speech-detector) speech-band energy share, truncation, tail handling and window statistics (mean, median, min, max, std, range, windows at/above threshold, saturated windows).
- Visual: frame size, face-area ratio, crop size, detector name, face selection rule, label map, statistics across frames.
- A **caveat** is attached to a branch when its windows/frames disagree strongly (range of 0.5 or more). It does not change any score.
- `scripts/diagnose_media.py <file> [--sensitivity]` prints all of the above for any file and re-scores it under documented preprocessing changes (other checkpoint, loudness normalisation, shifted window grid, different face-crop margins, whole frame instead of crop). The spread shows how fragile a score is; it is not a measure of correctness.

No threshold, weight, score or model was changed to make any example pass.

## 5. What would settle it

1. A **labelled** evaluation set that resembles the intended use (real and manipulated video/audio, similar codecs and recording conditions), kept separate from anything used to pick a threshold.
2. Per-branch ROC/AUC and the equal error rate on that set, then a threshold chosen on validation data and reported with its error rates; only then can "false positive rate at threshold t" be stated.
3. A comparison of aggregation rules (mean vs median vs max) and of fusion rules on the same labelled data, plus calibration if probabilities are to be shown.

See `docs/evaluation_plan.md`. Until that is done the application must keep presenting scores as raw, uncalibrated model outputs and "flagged" as "crossed an unvalidated default threshold".
