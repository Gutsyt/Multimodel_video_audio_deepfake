# Phase 0 findings

Status legend: **[V]** verified by me against the cited source or by running it;
**[V-2nd]** appears in a secondary source only (paper/loader/aggregator), not the primary one;
**[U]** unverified; **[H]** hypothesis that the real files must confirm or refute.
Nothing below was verified against actual FakeAVCeleb files, because none were available to me.

## 1. FakeAVCeleb (primary dataset candidate)

| Item | Finding | Status | Source |
|---|---|---|---|
| Access | Google request form; after acceptance the authors send a download-script link | [V] | Official repo README (DASH-Lab/FakeAVCeleb, raw.githubusercontent.com/DASH-Lab/FakeAVCeleb/main/README.md) |
| License | README: "data can be released under the FakeAVCeleb Request Forms"; code MIT. No standalone data-license text was seen. An aggregator (hyper.ai) lists "CC BY 4.0", which conflicts with the official wording | [V] request-form terms; **[U] actual terms**; aggregator claim disregarded | same; hyper.ai/en/datasets/19812 |
| Terms of underlying VoxCeleb2 | Not checked | [U] | – |
| Size per official README | 500 real, 19,500 fake, 20,000 total; 500 subjects; 4 methods; real and fake audio; "fine-grained labeling: Yes" | [V] (claim by the authors) | official README comparison table |
| Size per other sources | "21k videos from 500 identities" (DeepFake Doctor, arXiv 2506.05851); "FakeAVCeleb (21,544 videos)" (a third-party HF model card). Disagreement suggests dataset versions differ | [V-2nd] | arxiv.org/pdf/2506.05851; huggingface.co/vl4gh7/dslp-ensemble |
| **Usable sample counts** | **Cannot be determined without the files.** Do not use any number above in experiments | [U] | – |
| Categories | Four audio/video authenticity combinations (real/fake video x real/fake audio). Methods: FaceSwap, FSGAN (face), Wav2Lip (lip), SV2TTS (voice clone). Fakes are made from 500 real VoxCeleb2 videos | [V] | paper arXiv 2108.05080 (text retrieved via arxiv.org/pdf/2108.05080) |
| Category naming | Papers use inconsistent abbreviations (ARVR/AFVR vs RVFA/FVRA etc.). The real strings in the data must be read from the files | [V-2nd] | arXiv 2108.05080, 2310.13103, 2405.04097 |
| Official splits | The official README says nothing about train/val/test splits. Papers build their own identity-based splits (e.g. 430/70 subjects; 60/10/30 by identity) | [V] absence in README; [V-2nd] papers | README; arXiv 2311.02733, 2605.19630 |
| Pair construction | Face-swap pairs are chosen as the top-5 most similar faces by Face++ similarity, same gender/ethnicity group | [V] | arXiv 2108.05080 |
| Known shortcut | Fake-video/fake-audio samples reportedly contain a few ms of leading silence absent from real ones (attributed to Boldisor et al.) | [V-2nd] | arXiv 2506.05851 |
| Label-file structure | A third-party loader reads `meta_data.csv` with columns `source`, `method`, `type`, `filename`, takes `type.split("-")[-1]` as the audio label string ("FakeAudio"/"RealAudio"), and uses `method == "real"` for real videos. Folder `FakeAVCeleb-audio`, `.mp3` audio | [V-2nd], **[H]** | huggingface.co/spaces/ldhldh/streamlit_demo (src/datasets/fakeavceleb_dataset.py) |

### Hypotheses the inventory must confirm or refute (nothing is coded against these until then)
- H1 Directory layout and archive structure are as the third-party loader implies (unknown if official).
- H2 `meta_data.csv` exists with columns `source`, `method`, `type`, `filename` (and possibly others, e.g. target identity, gender, race). Actual column names and value vocabularies are unknown.
- H3 Total video count is 20,000 (official) vs ~21.5k (other sources); depends on dataset version.
- H4 One real video per identity; 500 identity IDs.
- H5 Mean duration about 7.8 s (aggregator claim only).
- H6 Every video carries an audio stream (inventory `stream_combinations` will show).
- H7 No official split file ships with the data.
- H8 Meaning of `source`/`target` for each method (which identity's face and which voice appear in a fake) must be read from the metadata, not assumed.
- H9 Resolution, fps, codecs, and sample rates are uniform (a uniformity itself can be a shortcut: see risks).
- H10 The leading-silence shortcut is present in the copy you receive.

## 2. Supplementary datasets

| Dataset | Finding | Status | Source |
|---|---|---|---|
| ASVspoof 2019 (LA, PA) | Rights: Open Data Commons Attribution License. LA.zip is 7.116 GB, PA.zip 16.44 GB. Partitioned into train/dev/eval with 20/10/48 speakers (LA and PA). LA has 17 TTS/VC systems (6 known attacks, 11 unknown) | [V] | datashare.ed.ac.uk/handle/10283/3336 (+ ?show=full); arXiv 1911.01601; ar5iv 1904.05441 |
| ASVspoof 2019 audio only | Yes: no video. Role: audio pretraining and cross-dataset audio test | [V] | same |
| FaceForensics++ | Per the FakeAVCeleb README comparison table, FF++ has no audio. Access terms, splits, and sizes were **not verified** in Phase 0 | [V-2nd] audio; [U] rest | FakeAVCeleb README |
| AV-Deepfake1M | Not verified in Phase 0; stretch goal only | [U] | – |

## 3. Synchronization model feasibility (kept OPTIONAL)

| Item | Finding | Status |
|---|---|---|
| Candidate | SyncNet (Chung & Zisserman, ACCV 2016 workshop) via `joonson/syncnet_python`; outputs AV offset, min distance, confidence; same checkpoint underlies Wav2Lip's LSE-D/LSE-C metrics | [V] |
| Code license | MIT (GitHub repo page) | [V] |
| Weights license | A PyPI repackaging quotes the Oxford VGG page: weights "can be used for research purposes under Creative Commons Attribution License" and says MIT/Apache do not cover weights. I could not reach robots.ox.ac.uk to read the primary statement | [V-2nd], **[U] primary** |
| Weights download in sandbox | `download_model.sh` host not reachable from my sandbox (robots.ox.ac.uk returned no connection) | [V] |
| Reference output | README demo on `data/example.avi` should print approx `AV offset: 3`, `Min dist: 5.353`, `Confidence: 10.021` | [V] (README claim) |
| I ran SyncNet | **No.** Neither weights nor face-detector weights were obtained; no sync score has been computed | [V] |
| Modern-Python status of the original repo | Unknown (forks/repackagings exist for modern Python; not evaluated) | [U] |

**Evaluation caveat:** FakeAVCeleb has authenticity labels, not sync labels. Wav2Lip forgeries are generated with a SyncNet-style discriminator, so they may look well-synchronized to SyncNet, and one third-party card notes FakeAVCeleb forgeries often leave mouth motion largely intact [V-2nd]. Sync scores therefore cannot be treated as a deepfake detector on this dataset without evidence.

**Gates before sync enters the project** (all must pass, otherwise it stays out):
- G1 Weights obtained from the primary source and their terms read and recorded.
- G2 Reference demo reproduced within tolerance on `example.avi` (offset 3, distance ~5.35, confidence ~10.0).
- G3 Controlled evaluation defined: take real clips, shift the audio by known offsets with FFmpeg, and measure offset-detection error. This evaluates the *sync tool*, never deepfake detection.
- G4 Runtime acceptable on CPU for 5-10 s clips; behavior defined when no face/speech (status UNRELIABLE, not a score).

## 4. Environment compatibility

| Check | Result | Status |
|---|---|---|
| Sandbox | Ubuntu 24.04, Python 3.12.3, 1 CPU, 3.9 GB RAM, ~10 GB free disk, no GPU | [V] |
| FFmpeg / ffprobe | 6.1.1-3ubuntu5 present; libx264 encode, AAC, ffprobe JSON used successfully | [V] |
| Python/tooling tested | Only Python 3.12.3. 3.10/3.11 are declared in `pyproject.toml` but **untested** | [V] / [U] |
| Non-ML stack resolves together on 3.12 | `pip install --dry-run` resolved: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0, opencv-python-headless 5.0.0.93, librosa 1.0.0, soundfile 0.14.0, fastapi 0.161.1, uvicorn 0.54.0, SQLAlchemy 2.1.6, alembic 1.20.1, pydantic-settings 2.16.1, python-multipart 0.0.32, httpx 0.28.1. Resolution only: **nothing was installed, imported, or run** | [V] resolution only |
| torch / torchvision / timm | Latest on PyPI: torch 2.14.1, torchvision 0.29.1, timm 1.0.30. A first full dry-run was **inconclusive** (timed out while downloading wheels). Declared metadata shows torch's Linux wheel depends on CUDA 13 packages (cuda-toolkit, cudnn, nccl, triton), i.e. a multi-GB install. **Not installed or tested here** | [V] metadata; [U] runtime compatibility with numpy 2.x/opencv 5 |
| CPU-only torch wheel | `download.pytorch.org` is unreachable from my sandbox, so I could not test it | [U] |
| `facenet-pytorch` 2.6.0 | Declares `torch<2.3`, `torchvision<0.18`, `numpy<2`, `Pillow<10.3`: conflicts with the current stack. **Not recommended**; choose another face detector in Phase 2 (candidates, all untested: OpenCV YuNet via `cv2.FaceDetectorYN`, MediaPipe, RetinaFace implementations) | [V] metadata |
| Pretrained weights hosts | `download.pytorch.org` and `huggingface.co` unreachable from my sandbox; PyPI, GitHub, raw.githubusercontent.com reachable. Reachability from **your** machine is unknown | [V] sandbox; [U] yours |
| FakeAVCeleb's own code env | Pins Python 3.8, torch 1.8, numpy 1.20: do not reuse | [V] |
| Hardware for the plan | This sandbox cannot train. Training needs your GPU or Colab/Kaggle; whether their free tiers fit is **unverified** | [U] |

## 5. Risks added or sharpened by Phase 0
1. **Identity-linkage leakage:** each fake pairs a source with one of its 5 most similar targets. The identity-pair graph may form one giant connected component, making strict group splits impossible. Decide with real metadata in Phase 1; if needed, document a fallback and the residual leakage.
2. **Version ambiguity:** 20,000 vs ~21.5k videos; record the dataset version and checksums.
3. **Shortcuts:** leading silence [V-2nd]; possibly uniform codec/resolution/duration differences between real and fake (H9). Perturbation and silence-trimming checks are added to Phases 3 and 6.
4. **Label vocabulary inconsistency** across papers: never import label mappings from papers; derive from the files.
5. **Licensing:** access terms are request-form based; no media in Git; VoxCeleb2 terms unchecked.
6. **Sync on this dataset:** may not discriminate (see section 3).
7. **Compute:** your GPU situation is unknown; the sandbox is not a training environment.

## 6. Remaining blockers (need you)
- B1 Submit the FakeAVCeleb request form and receive the download link (see `dataset_access.md`).
- B2 Run `inspect_dataset.py` on the real data and send me `INVENTORY.md` (and `inventory.json` if possible).
- B3 Tell me your hardware (OS, GPU, RAM, free disk) via `check_environment.py --json`.
- B4 Confirm whether your machine can reach `download.pytorch.org` / `huggingface.co` (needed for pretrained weights).
- B5 Read the request-form terms yourself and confirm no restriction conflicts with the intended use (research paper, local demo).

## 7. Precise acceptance criteria for Phase 1 (Manifest, labels, splits)
Phase 1 starts only after B2 is resolved and you approve these findings.
1. **Inputs:** the real `inventory.json`; every hypothesis H1-H10 is marked confirmed/refuted/unresolved in `docs/dataset_facts.md`, with the inventory line that proves it.
2. **Manifest schema** (versioned, tested): `media_id` (unique), relative `video_path`, `audio_path`/extraction status, `video_status` and `audio_status` (values of `ModalityStatus`), `label_video`, `label_audio` in {real, fake, unknown}, `manipulation_method` (from metadata else `unknown`), identity fields exactly as present in the metadata, `split`, probe facts (duration, resolution, fps, sample rate), `label_provenance` (the column/rule that produced each label), `notes`.
3. **No silent labels:** a label is set only by an explicit, documented rule from an explicit metadata field; anything else is `unknown`, counted, reported, and excluded from training/evaluation. A test with deliberately ambiguous fixture rows proves this.
4. **Cross-checks:** where both metadata and path/filename conventions exist, disagreements are listed in a report (not auto-resolved).
5. **Splits:** reproducible from a recorded seed + config hash; grouping rule justified from the real identity linkage; automated leakage tests pass (no group in two splits; no byte-identical duplicate across splits; derived segments inherit their video's split). If a giant component exists, the chosen fallback and its residual leakage are documented.
6. **Report:** per split and per category (real/fake x audio/video), per method, missing/corrupt counts, duration/resolution stats, all generated by script from the files; no hand-typed numbers.
7. **Quality gates:** `ruff`, `mypy --strict`, and `pytest` pass; new tests use synthetic fixtures only and are labeled as software tests.
8. **Hygiene:** no dataset media or inventory dumps committed (`.gitignore` verified); commands and run directories documented.
9. **Out of scope for Phase 1:** frame/audio extraction, any model code, sync, fusion, API, frontend.
