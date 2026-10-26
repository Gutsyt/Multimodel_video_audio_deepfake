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

## 6. Follow-up with labelled samples (26 Oct 2026, later)
Public labelled samples were used: **73 genuine LibriSpeech read-speech clips** (human recordings), **180 real LFW photos**, **120 real FFHQ faces** and **100 StyleGAN faces** (the "140k real and fake faces" data, fake half). They are small and from specific domains, so the numbers describe those samples only. `scripts/evaluate_labelled.py` reproduces them on your own labelled files.

**Ruled out**
- Vendored AASIST code and both weight files are byte-identical to upstream.
- Audio codec: the same 40 genuine clips flagged 1/40 as FLAC, AAC 128k, AAC 64k and MP3 64k.
- Window-edge clicks: fading window edges by 10 or 50 ms changed nothing.
- Silence: windows with more leading/trailing silence were flagged *less*, not more.

**Found and fixed: tiled trailing windows (a logic flaw)**
- The old code looped a short trailing piece of a longer clip to fill a 4.04 s window. On the genuine clips, natural windows reached 0.5 in 4/63 cases but tiled windows in **20/41**, and that window counted equally in the mean.
- Clip-level result on 73 genuine clips: 3 flagged before, **1 after**, mean score 0.141 to 0.088. Through the live API on 10 clips that were previously affected: 3 flagged before, 1 after.
- The trailing window is now the last 4.04 s of the clip (real audio, overlapping the previous window). Clips shorter than one window are still looped, as upstream does for short utterances.
- Caveat: the window is counted once more in the mean although it adds less than a full window of new audio.

**Found, not fixable by logic**
- Remaining audio flags: 10 of 120 natural windows (8%) on clean read speech are scored as spoof; they are mostly windows cut through the middle of speech. That is the model's own error rate on this sample. A 1963 speech recording scored 0.99 in two windows (domain shift).
- **Visual branch is at chance on matched data**: AUC 0.50 to 0.55 for real FFHQ vs StyleGAN faces. Roughly 43 to 47% of real FFHQ faces score at or above 0.5 (12/28 as an image and as H.264 video), 94% of real LFW photos did on the whole image, and heavy H.264 compression (CRF 38) raised the real-face flag rate to 18/29. Which side of 0.5 a face lands on depends mainly on crop margin and image quality (13% to 52% of real faces flagged across margins), not on whether it is fake. The first LFW-vs-StyleGAN comparison (AUC 0.13) was a data-domain confound; the matched comparison removes it. **A visual-branch flag is therefore weak evidence, and a genuine video being flagged by it is expected.**
- Haar false boxes: 3 of 15 face-free images (coins, moon, star field) produced a small "face". A persistent false box in a face-free video could pass the frame-count gate.

**Decision logic made explicit**
- The overall status is unchanged (fused score at or above the threshold), but when the two branches land on opposite sides the assessment now carries `branches_agree: false` and the summary says it rests on an unvalidated average. Example from the live API: audio 0.67 (flagged) with visual 0.37 (not flagged) fused to 0.52 and was reported flagged.
- No threshold, weight, crop margin or score was tuned. Without labelled spoofed audio and a visual model that separates the classes there is no operating point to choose.
