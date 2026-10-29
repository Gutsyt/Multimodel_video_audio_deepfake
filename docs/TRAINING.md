# Training guide

This guide trains the two detectors the app needs and measures them. When it's done there are two small files, `src/avdd/assets/visual_head.npz` and `audio_head.npz`. Until they exist the app starts but refuses to analyse, with the message *"The trained model files are missing"*.

Written for whoever runs the training (Gutsy). No prior ML experience is assumed: every command says what it does and what a good result looks like.

---

## 0. The idea in one minute

- Two large networks, pretrained by OpenAI and Meta, are used **frozen** (never changed):
  - **CLIP ViT-L/14** for faces
  - **XLS-R 300M** for speech
- Each one turns a face crop or a 4-second speech window into a list of 1,024 numbers, an **embedding**.
- A tiny **logistic-regression classifier** learns which embeddings come from real media and which from fakes.
- The classifier's output is then **calibrated**, so 0.8 really means about 80% likely fake.
- Two **thresholds** are set on held-out data:
  - *manipulated*: at most about 2% of genuine media reaches it
  - *authentic*: about 90% of genuine media falls at or below it

Why freeze the big networks? Fine-tuning them on a few thousand examples overfits to the specific fakes seen. Frozen features plus a linear classifier is what research shows generalises best to new fakes and new recording conditions (Ojha et al., CVPR 2023; Pascu et al., Interspeech 2024). It is also cheap: the expensive part is just running the frozen networks once over the data.

```
download data ─► embed faces + speech (GPU-heavy) ─► fit classifiers ─► test on held-out files ─► report
   10–30 min          1–2 h on GPU / ~3.5 h CPU        2–5 min              15–40 min             1 min
```

---

## 1. Where to train: pick one

| Platform | Cost | GPU | Good for | Watch out |
|---|---|---|---|---|
| **Google Colab (recommended)** | free | NVIDIA T4, 15 GB | easiest: open the notebook, click Run all | keep the tab open (idle disconnect after ~90 min); daily GPU quota of a few hours, more than enough for one run |
| Kaggle Notebooks | free (phone verification) | T4 ×2 or P100, 30 h/week | runs in the background after *Save & Run All*, no tab needed | internet and GPU must be switched on in notebook settings |
| Your own laptop with an NVIDIA GPU (≥ 6 GB VRAM) | free | yours | no time limits; data stays local | needs CUDA PyTorch; 25 GB free disk |
| Your own laptop, CPU only | free | none | works anywhere | slow: about 3.5 h for the embedding step; 8 GB RAM minimum |

**Recommendation: Google Colab.** Zero setup, a free GPU, and the notebook below runs everything. The total run fits easily in one free session. Progress is saved to Google Drive, so a disconnect costs nothing: reconnect and run again.

---

## 2A. Train on Google Colab (recommended)

1. Open **https://colab.research.google.com** → *File → Open notebook → GitHub* tab → paste
   `https://github.com/Gutsyt/Multimodel_video_audio_deepfake` → choose branch **v2-update** → open **`notebooks/train_colab.ipynb`**.
2. *Runtime → Change runtime type → **T4 GPU*** → *Save*.
3. Optional, for faster downloads: create a free token at https://huggingface.co/settings/tokens (type *Read*). In Colab, click the key icon on the left → *Add new secret* → name `HF_TOKEN`, paste the value, and enable *Notebook access*.
4. *Runtime → **Run all***. When asked, allow access to Google Drive (it stores progress in `MyDrive/avdd_cache`).
5. Keep the tab open. Check back about every 30–60 min.
6. When cell 9 shows the graphs, go to **section 4** below to check the numbers, then send the results back (cell 11, or **section 5**).

If the session disconnects: *Runtime → Run all* again. Finished downloads and embedding sources are skipped automatically.

What each notebook cell does:

| Cell | What it does | Time on T4 |
|---|---|---|
| 1 | shows the GPU (should say Tesla T4) | instant |
| 2 | mounts Google Drive and sets the folders `DATA`, `CACHE`, `E2E` | instant |
| 3 | clones the repo, branch `v2-update` | 10 s |
| 4 | installs the project (`pip install -e ".[video,hf,train]"`) | 1–2 min |
| 5 | downloads the datasets (section 3.1) | 10–30 min |
| 6 | face embeddings (section 3.2) | 30–60 min (video decoding is the bottleneck) |
| 6b | speech embeddings | 10–20 min |
| 7 | fits classifiers, calibration and thresholds; writes the two `.npz` files | 2–5 min |
| 8 | builds 383 held-out test files and runs them through the real app | 15–30 min |
| 9 | draws the graphs and fills the README results | 1 min |
| 10 | optional wiring test with the real models | 2 min |
| 11 | pushes the results to GitHub, or downloads a zip | 1 min |

---

## 2B. Train on your own laptop

### Windows (PowerShell)
```powershell
# one-time tools
winget install Git.Git Python.Python.3.12 Gyan.FFmpeg OpenJS.NodeJS.LTS
# close and reopen PowerShell, then:
git clone -b v2-update https://github.com/Gutsyt/Multimodel_video_audio_deepfake.git
cd Multimodel_video_audio_deepfake
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# PyTorch: NVIDIA GPU → CUDA build; no GPU → CPU build (pick ONE line)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[video,hf,api,train]" -r requirements-dev.txt
python -c "import torch; print('GPU:', torch.cuda.is_available())"

$env:DATA="$HOME\avdd_data"; $env:CACHE="$HOME\avdd_cache"; $env:E2E="$HOME\avdd_e2e"
$env:HF_TOKEN="hf_..."                    # optional
$DEV = "cuda"                             # or "cpu" if GPU: False
```

### Linux / macOS (bash)
```bash
sudo apt install -y git python3-venv ffmpeg        # macOS: brew install git python ffmpeg node
git clone -b v2-update https://github.com/Gutsyt/Multimodel_video_audio_deepfake.git
cd Multimodel_video_audio_deepfake
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA; CPU: .../whl/cpu; macOS: plain `pip install torch`
pip install -e ".[video,hf,api,train]" -r requirements-dev.txt
export DATA=~/avdd_data CACHE=~/avdd_cache E2E=~/avdd_e2e
export HF_TOKEN=hf_...                              # optional
DEV=cuda                                            # cpu without NVIDIA; mps on Apple Silicon
```

Then run section 3. In PowerShell, write `$env:DATA` where the commands say `$DATA`, and `$DEV` stays as is.

**CPU only:** before the embedding step run `export OMP_NUM_THREADS=4` (PowerShell `$env:OMP_NUM_THREADS=4`), and keep the laptop plugged in with sleep disabled.

---

## 3. The steps explained (same commands everywhere)

### 3.1 Download the data: `scripts/download_data.py`
```bash
python scripts/download_data.py $DATA
```
| Dataset | Why it's in the mix | Amount |
|---|---|---|
| FaceForensics++ c23 | the standard benchmark: 1,000 real YouTube videos + 6 manipulation methods, compressed like real uploads | 1,000 real + 1,020 fake videos (read out of an 18 GB zip, only these files) |
| Celeb-DF / FakeAVCeleb frames | high-quality face swaps and lip-sync (Wav2Lip) fakes | all real frames + 2–3% of fakes |
| WildDeepfake | deepfakes collected from the internet, real-world quality | 44 archive files |
| ASVspoof 2019 LA, ASVspoof 5 | voice cloning / TTS attacks; ASVspoof 5 adds codecs and compression | 2 shards each |
| In-the-Wild | real recordings of 58 public figures vs deepfakes of them | complete |

**Success:** the last line is `DONE`, and the `DATA` folder is about 10 GB. Interrupted? Run it again; it resumes.

### 3.2 Embed: `scripts/train_heads.py embed-visual` / `embed-audio`
```bash
python scripts/train_heads.py embed-visual --data $DATA --cache $CACHE --device $DEV
python scripts/train_heads.py embed-audio  --data $DATA --cache $CACHE --device $DEV
```
- **Faces:** from each FF++ video 2 frames are taken, the face is found with YuNet and cropped exactly as the app does at runtime, and CLIP features are stored for layers 12, 16, 20 and 24.
- **Speech:** a 4 s window from each clip, XLS-R features averaged over time, stored for layers 4, 6, 8, 10 and 12.
- **Splits** are decided by a hash of the person, video or speaker: 70% train, 15% validation, 15% test. The same person never appears in two splits, so the test numbers are honest.
- Output in `$CACHE`, one file per source:
  `vis_ff, vis_wdf, vis_wdf_test, vis_celeb, vis_celeb_test, aud_ASVspoof2019_LA, aud_ASVspoof5, aud_InTheWild` (`.npz`)
- **Progress** looks like `vis: 1088 embedded (25.3/s)`; `saved vis_ff 4021` means a source is finished. Re-running skips finished sources.

Options: `--frames 3` gives more face frames per video (better and slower); `--audio-caps "ASVspoof2019_LA=3000,ASVspoof5=3000,InTheWild=7000"` gives more speech.

### 3.3 Fit: `scripts/train_heads.py fit`
```bash
python scripts/train_heads.py fit --cache $CACHE | tee fit_log.txt
```
For each check it tries every stored layer × 3 regularisation strengths, keeps the best validation AUC, calibrates (Platt scaling) and sets the thresholds. It then evaluates the test sets and writes `src/avdd/assets/visual_head.npz` and `audio_head.npz`. Each file carries the weights, calibration, thresholds and all metrics, which the app shows on its Models page.

### 3.4 Test end to end: `scripts/build_test_sets.py` + `scripts/evaluate_folder.py`
```bash
python scripts/build_test_sets.py $DATA $E2E
mkdir -p e2e_results
python scripts/evaluate_folder.py --device $DEV --real $E2E/A_ff/real  --fake $E2E/A_ff/fake  --json e2e_results/A_ff.json
python scripts/evaluate_folder.py --device $DEV --real $E2E/B_wdf/real --fake $E2E/B_wdf/fake --json e2e_results/B_wdf.json
python scripts/evaluate_folder.py --device $DEV --real $E2E/C_itw/real --fake $E2E/C_itw/fake --json e2e_results/C_itw.json
python scripts/evaluate_folder.py --device $DEV --real $E2E/D_av/real                         --json e2e_results/D_av.json
```
These are real **files** from test-split people only, run through the exact code the web app uses:

| Set | What it contains |
|---|---|
| A | FF++ videos |
| B | WildDeepfake clips |
| C | In-the-Wild speech |
| D | genuine video with genuine speech; tests the combined false-alarm rate when both checks run |

Each command ends with a summary such as `real: n=60 likely_authentic=51 (85.0%), inconclusive=8, likely_manipulated=1 (1.7%)`.

### 3.5 Report: `scripts/make_report.py`
```bash
python scripts/make_report.py --cache $CACHE --e2e e2e_results
```
Writes `docs/figures/*.svg` (data, layer search, ROC curves, score distributions, end-to-end verdicts) and replaces the Results section of `README.md`. Every number comes from the files above; nothing is typed by hand.

---

## 4. Is the training good? Check these numbers

Look at `fit_log.txt`, or the README Results table after 3.5.

| What | Where | Good | Acceptable | Problem |
|---|---|---|---|---|
| Face AUC, held-out test | `visual … in_domain_test.auc` | ≥ 0.90 | 0.85–0.90 | < 0.85 |
| Face AUC, WildDeepfake (hardest) | `wilddeepfake_test.auc` | ≥ 0.80 | 0.70–0.80 | < 0.70 |
| Voice AUC, held-out speakers | `audio … in_domain_test.auc` | ≥ 0.95 | 0.90–0.95 | < 0.90 |
| **Genuine flagged** (false alarms), each test set | `genuine_flagged` | ≤ 0.02 | ≤ 0.05 | > 0.05 |
| **Genuine files flagged end to end** | `real: … likely_manipulated=` in 3.4 | ≤ 3% | ≤ 5% | > 5% |
| Fakes flagged | `fake_flagged` | ≥ 0.60 | 0.40–0.60 | < 0.40 |

*Fakes flagged* is lower than AUC suggests. That's on purpose: the threshold favours **not** accusing genuine media, and borderline fakes land in *inconclusive*.

If something is in *Problem*: don't edit thresholds by hand. Note it, send the results anyway (section 5), and see section 6.

Also test your own media, which is the real proof. Put 20 or more genuine videos (phone clips, calls) in a folder and run:
```bash
python scripts/evaluate_folder.py --device $DEV --real path/to/my_videos --json e2e_results/mine.json
```

---

## 5. Send the results back

Files to deliver (all small):
- `src/avdd/assets/visual_head.npz` and `src/avdd/assets/audio_head.npz` (**required**: the app works only with these)
- `fit_log.txt`, `e2e_results/*.json`, `docs/figures/*.svg`, `README.md`

**Option A: git push** (from a laptop, or Colab cell 11 with a `GH_TOKEN` secret):
```bash
git add src/avdd/assets/*_head.npz docs/figures README.md
git add -f fit_log.txt e2e_results/*.json
git commit -m "Trained heads and measured results"
git push origin v2-update
```
The open pull request for `v2-update` then updates by itself.

**Option B: upload on GitHub:** open the repo → branch `v2-update` → *Add file → Upload files*. Drag the files in, keeping their folders (`src/avdd/assets/`, `docs/figures/`), and commit.

Then the app works for everyone who pulls the branch. Check with:
```bash
python -m pytest -q -m real_models      # loads the real models; should pass
python scripts/run_server.py            # open http://127.0.0.1:8000 → header shows "Models ready"
```

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| Colab: `nvidia-smi` not found / no GPU | *Runtime → Change runtime type → T4 GPU*, then run all again |
| Colab: "cannot connect to GPU backend" | the free GPU quota for today is used up; try again tomorrow or use Kaggle |
| Colab disconnected mid-run | reconnect → *Run all*; finished work is skipped (cache on Drive, downloads re-checked) |
| `HTTP 429` or very slow downloads | add the `HF_TOKEN` secret / variable, wait a few minutes, re-run |
| `CUDA out of memory` | another notebook is using the GPU: *Runtime → Disconnect and delete runtime*, then run all |
| Laptop: process killed / freezes | close other apps; run the two embed commands one after the other |
| `ffmpeg not found` | install FFmpeg (see 2B) and reopen the terminal |
| `ModuleNotFoundError: avdd` | activate the venv and run `pip install -e ".[video,hf,api,train]"` from the repo folder |
| A metric is in *Problem* | send the results anyway, with a note; more data (`--frames 3`, larger `--audio-caps`, more WildDeepfake tars via `download_data.py --wdf-tars 24`) usually helps |
