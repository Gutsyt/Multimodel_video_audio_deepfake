# AVDD lens: audio-visual deepfake detection

Upload a **video, voice clip or photo**. Two independent checks run on it: one judges the **face**, one judges the **voice**. Each check is compared with thresholds set on held-out data, and the app returns one of four verdicts: **likely authentic**, **inconclusive**, **likely manipulated**, or **not enough to judge** (no usable face or speech).

The previous version flagged genuine videos as deepfakes. Its own investigation showed why: the face model scored at chance (AUC 0.50–0.55) and flagged 43–94% of real faces, while the voice model (AASIST) was trained only on clean studio speech. Version 1.0 replaces both models, trains them on real-world data, and sets the thresholds so that genuine media is rarely flagged. The rate is measured below.

## How a file is judged

```
upload ─► ffprobe (what is in the file?)
          ├─ video / image ─► up to 12 frames ─► YuNet face detector ─► face crop
          │                    ─► CLIP ViT-L/14 (frozen) ─► linear head ─► P(fake) per frame ─► median
          └─ audio track   ─► 16 kHz mono, 4 s windows ─► wav2vec 2.0 XLS-R 300M (frozen, 12 layers)
                               ─► linear head ─► P(fake) per window ─► median
verdict:  any check ≥ its t_fake            → likely manipulated
          every check ≤ its t_real          → likely authentic
          otherwise                         → inconclusive
          no usable face or speech          → not enough to judge
```

**Why "any check" flags:** a fake can be one-sided, such as a cloned voice over a real video or a face swap over the real voice. Each check's `t_fake` is set so that only about 2% of genuine *validation* media reaches it. The rates actually measured on held-out test data are in the Results section.

**Why the median:** one bad frame (motion blur, side profile) or one noisy window cannot flip the result on its own.

## The models and why these

| Check | Frozen backbone | What is trained here | Recipe from |
|---|---|---|---|
| Face | `openai/clip-vit-large-patch14` (CLS token of one hidden layer, chosen on validation data) | logistic regression + Platt calibration | Ojha et al., *Towards Universal Fake Image Detectors*, CVPR 2023; Khan & Dang-Nguyen, *CLIPping the Deception*, 2024 |
| Voice | `facebook/wav2vec2-xls-r-300m`, first 12 transformer layers (time-averaged hidden state of one layer, chosen on validation data) | logistic regression + Platt calibration | Pascu et al., *Towards generalisable and calibrated audio deepfake detection with self-supervised representations*, Interspeech 2024 |
| Face finder | YuNet (OpenCV Zoo, MIT, 230 KB, vendored) | none | replaces the old Haar cascade, which produced false boxes |

Fine-tuning a whole network on a few thousand samples overfits to the specific fakes it saw. A linear probe on frozen foundation-model features is the approach the papers above show generalises best to unseen generators and recording conditions. It is also the only one trainable on a CPU in hours. Each trained head is a 1024-weight vector shipped in `src/avdd/assets/*_head.npz` (a few KB), and inference is a single dot product.

## Training data

All datasets come from Hugging Face. They were downloaded only partly, by shard or by HTTP range reads, and are never committed. Splits are by **identity**, using a hash of the video ID / sequence / speaker: 70% train, 15% validation (calibration and thresholds), 15% test. No person in the test split appears in training.

| Source | What it adds | How much was used |
|---|---|---|
| [FaceForensics++ c23](https://huggingface.co/datasets/bitmind/FaceForensicsC23) | 1,000 real YouTube videos + 6 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, Google DeepFakeDetection), H.264-compressed | all 1,000 real + 170 per method; 2 frames per video, cropped by the app's own YuNet code |
| [Celeb-DF / FakeAVCeleb frames](https://huggingface.co/datasets/thenewsupercell/celeb-df-image-dataset) | celebrity interview face-swaps and lip-sync (Wav2Lip) fakes | every real frame, plus about 2–3% of fake frames (fakes outnumber reals 80:1) |
| [WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake) | deepfakes collected from the internet, real-world quality | 12 + 12 train tars, 10 + 10 test tars, 2 frames per sequence |
| [ASVspoof 2019 LA](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) | studio speech vs 13 TTS / voice-conversion systems | 2,000 random clips |
| [ASVspoof 5](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5) | newer TTS / VC attacks, **codec and compression conditions** | 2,000 random clips |
| [In-the-Wild](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild) | real recordings of 58 public figures vs deepfakes of them (interviews, speeches, phone quality) | 4,500 random clips, split by speaker |

The training inputs pass through **exactly the inference code**: `sample_frames` → `YuNetFaceDetector` → `crop_face_rgb` → the CLIP processor, and `make_windows` → the XLS-R feature extractor. A mismatch between training and inference preprocessing is a common hidden cause of false alarms, and this rules it out.

## How it was trained

```bash
python scripts/download_data.py DATA                                    # ~10 GB, see docs/RUNBOOK.md
python scripts/train_heads.py embed-visual --data DATA --cache CACHE   # frozen CLIP features per face crop
python scripts/train_heads.py embed-audio  --data DATA --cache CACHE   # frozen XLS-R features per 4 s window
python scripts/train_heads.py fit --cache CACHE                         # heads, calibration, thresholds
python scripts/make_report.py --cache CACHE --e2e E2E_JSON_DIR          # figures + the Results section below
```

1. **Embed:** every sample becomes a 1024-dim feature. Face: CLS token of CLIP layers 12, 16, 20 and 24. Voice: mean over time of XLS-R layers 4, 6, 8, 10 and 12. All are stored as float16.
2. **Fit:** for each layer and each regularisation strength C ∈ {0.01, 0.1, 1}, fit an L2-normalised, standardised, class-balanced logistic regression on the train split. Keep the best validation AUC.
3. **Calibrate:** fit Platt scaling on the validation logits, so a score of 0.8 means roughly an 80% chance of being fake on data like the validation set.
4. **Thresholds:** on validation, aggregated per video for faces and per clip for voices:
   - `t_fake` is set just above the 98th percentile of genuine scores, so at most 2% of genuine media is flagged. It is never below 0.5, so a file the calibrated model thinks is more likely real is never flagged.
   - `t_real` is the 90th percentile of genuine scores, so about 90% of genuine media is cleared as authentic.
5. **Test:** report AUC, EER and verdict rates on the test split and on separate test sets (WildDeepfake test tars, Celeb-DF test frames). Then run 383 held-out **files** through the full app (`scripts/evaluate_folder.py`).

## Resources used

| | |
|---|---|
| Hardware | 4 vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM, **no GPU** (cloud container) |
| Software | Python 3.13, PyTorch 2.15 (CPU), Transformers 5.21, scikit-learn 1.9, OpenCV 4.14, FFmpeg |
| Downloaded | about 6 GB datasets (parquet / tar shards) + 4.2 GB FF++ videos (2,020 files via range reads) + 2.9 GB backbones |
| Feature throughput (2 threads per job, both jobs in parallel) | faces: about 0.9 crops/s (ViT-L/14); speech: about 1.75 windows/s (12-layer XLS-R) |
| Head fitting | minutes; logistic regression on ≤10k × 1024 features |
| Peak memory | about 2.5 GB (face job), about 1.8 GB (voice job), with streaming in batches of 32 |

Inference on your machine: about 4 GB RAM for both backbones. A 10 s video takes roughly 10–20 s on a 4-core CPU, and is near-instant on a GPU (`AVDD_DEVICE=cuda`).

## Results

<!-- RESULTS:START (generated by scripts/make_report.py; do not edit by hand) -->
*Pending: written by `scripts/make_report.py` from the finished training run.*
<!-- RESULTS:END -->

## Run it

**Step-by-step training, evaluation and a full feature checklist: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).**

Prerequisites: Python 3.10+, Node 20+, and **ffmpeg + ffprobe on PATH**.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[video,hf,api]" -r requirements-dev.txt
cd frontend && npm ci && npm run build && cd ..
python scripts/run_server.py                              # open http://127.0.0.1:8000
```

The first start downloads the two backbones (about 2.9 GB) and takes a few minutes; the header shows **Models ready** when the app is ready.

| Tool | Command |
|---|---|
| One file, verdict plus per-frame and per-window scores | `python scripts/diagnose_media.py clip.mp4` |
| **Check false alarms on your own media** | `python scripts/evaluate_folder.py --real my_real_videos/ --fake my_fakes/` |
| Retrain / extend with more data | see *How it was trained* |
| Tests and lint | `python -m pytest -q -m "not real_models"`, `ruff check .`, `mypy`, `cd frontend && npm test` |

## The app

- **Analyze:** drop a file and watch each step as it runs.
- **Result:**
  - the verdict, with an evidence scale showing where each check landed relative to its own thresholds
  - for files analysed in this tab: the video replayed with the scored face boxed frame by frame, a filmstrip of the scored face crops, and face/voice timelines with a real waveform, synced to the playhead
  - compare with an earlier analysis; JSON, PDF and share-card exports
- **History:** filter by verdict, delete entries.
- **Models:** the measured metrics of the loaded heads.

## Privacy

The upload exists only as a temporary file while its analysis runs and is deleted afterwards. Only the result JSON is stored locally (no media, no file name). The in-page replay uses the copy already in your browser tab and is never re-uploaded. Nothing is sent to third parties except the one-time model download from Hugging Face. There is no authentication, so don't expose the server publicly (see `docs/deployment.md`).

## Limits

- Results are **evidence, not forensic proof**.
- The face check judges single frames; it has no lip-sync or temporal model.
- New generators and heavy filters can evade it.
- The voice check judges speech only; music and silence are marked unusable.
- Error rates on media unlike the training sources can differ from the figures above. Measure your own with `evaluate_folder.py`.

Licences and attributions: `ATTRIBUTIONS.md`.
