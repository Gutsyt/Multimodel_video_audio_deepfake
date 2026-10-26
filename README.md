# AVDD: audio-visual deepfake analysis (research prototype)

**Status (26 Oct 2026): local prototype with a working backend and frontend. Pretrained-model INFERENCE only: no training, no performance evaluation, no accuracy claims.**
Scores are raw model outputs: uncalibrated and not validated on any dataset by this project. Output is not forensic or legal proof. "Not flagged" never means "authentic".

## What the application does
Upload a **video**, an **audio file** or an **image**. The backend decides what it can analyze by parsing the content, then runs only the branches that have usable evidence:

| Upload | Audio branch | Visual branch | Result |
|---|---|---|---|
| Video with usable audio and a detectable face | runs | runs | both scores, plus a fused score by the documented (unvalidated) rule |
| Video, no usable audio, face found | missing / unreliable | runs | visual-only: that branch's own score and decision, **no fused score** |
| Video with audio, no detectable face | runs | unreliable | audio-only: that branch's own score and decision, **no fused score** |
| Video with neither | not usable | not usable | `insufficient_evidence` |
| Audio file (WAV, MP3, FLAC, OGG, M4A, AAC) | runs | not in file | audio-only |
| Image (JPEG, PNG, WebP, BMP) | not in file | runs if a face is found | visual-only, or `insufficient_evidence` |
| Empty, corrupted, truncated or unsupported file | none | none | clear validation error (400 / 415 / 422); nothing is analyzed |

A missing or unusable branch is never replaced by 0, 1 or a default prediction. Overall status is `flagged`, `not_flagged` or `insufficient_evidence`. **Fusion rule**: only when two branches are usable, `fused = w_audio * audio + w_visual * visual` with weights renormalised (default 0.5 / 0.5). The weights, the 0.50 threshold and the whole rule are **unvalidated defaults**.

Why was a real video flagged? See `docs/false_positive_investigation.md`: both score mappings were verified (not inverted), diagnostics were added, and what can and cannot be concluded without labelled data is spelled out.

## Quick start (Windows PowerShell)
Prerequisites: Python 3.10+, Node 20+, and **ffmpeg + ffprobe on PATH** (`winget install Gyan.FFmpeg`, then reopen PowerShell; check with `ffmpeg -version`).
```powershell
# 1. Python environment (run from the repository root)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[video,hf,api]"
pip install -r requirements-dev.txt

# 2. Verify the pretrained models on this machine (the visual model downloads once from Hugging Face)
python scripts/probe_aasist.py
python scripts/probe_video_model.py --image .\face.jpg

# 3. Build the frontend (the backend serves frontend/dist)
cd frontend
npm ci
npm run build
cd ..

# 4. Run the application, then open http://127.0.0.1:8000
python scripts/run_server.py
```
If script execution is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. The first start takes a minute while models load; the UI shows live model state.

### Frontend development server (hot reload)
```powershell
python scripts/run_server.py                  # terminal 1: backend on :8000
cd frontend; npm run dev                      # terminal 2: UI on http://localhost:5173 (proxies /api)
```
To point a built UI at a backend on another origin: `$env:VITE_API_BASE = "https://api.example.org"; npm run build`, and start the backend with `$env:AVDD_CORS_ORIGINS = "https://app.example.org"`. See `frontend/.env.example`, `.env.example` and `docs/deployment.md`.

### Inspect one file branch by branch
```powershell
python scripts/diagnose_media.py .\clip.mp4                 # scores, logits, per-window/frame values, input metrics
python scripts/diagnose_media.py .\clip.mp4 --sensitivity   # re-score under documented preprocessing changes
```

## Tests, lint and type checks
```powershell
python -m pytest -q                       # backend: test doubles + real-model smoke tests (skipped if unavailable)
python -m pytest -q -m "not real_models"  # backend without the real models
ruff check .
mypy
cd frontend
npm run typecheck
npm test
npm run build
```
**Mocked vs real.** Almost every test uses stand-in detectors that return fixed numbers (`tests/stubs.py`; their model cards say `is_test_fixture`) and synthetic media, so they verify behaviour (which branches run, validation, fusion, UI wording) and say nothing about detection quality. `tests/real/` loads the real AASIST and Hugging Face models and checks plumbing only (models load, outputs finite, label mapping matches the checkpoint config). Frontend tests use hand-written fixtures and a mocked `fetch`; `src/e2e.test.ts` (skipped unless `AVDD_E2E_URL` and `AVDD_E2E_VIDEO` are set, see `docs/runbook_local.md`) drives the real client against a running backend.

Known, pre-existing: `tests/unit/test_inventory.py` and `scripts/inspect_dataset.py` import `avdd.data.inventory`, which was never committed (the `data/` rule in `.gitignore` excludes `src/avdd/data/`); the test is skipped, and `ruff` reports the file-mode warning EXE001 for scripts with a shebang, and `mypy` reports two errors (that missing module and an untyped `from_pretrained` call).

## Architecture
`src/avdd/{media,audio,video,fusion,pipeline,api,common}`; `third_party/aasist` (vendored, MIT); `frontend/` (React 19 + Vite + TypeScript, no router or chart library; charts are hand-built SVG from real API data); `scripts/`; `tests/`.
Upload (raw body) -> ffprobe validation by content (kind: video / audio / image) -> queued job -> `analyze_media` (audio branch, visual branch, fusion, assessment) -> result in SQLite (no media, no file name) -> UI.

## Privacy and retention
The uploaded file exists only as a randomly named temporary file while its job runs and is deleted when the job ends (success or failure); leftovers are swept at startup. Only the result JSON is stored, locally, until you delete it (per record, or all finished records, from History). Nothing is sent to third parties except the one-time model download from Hugging Face. The server has no authentication: do not expose it to untrusted networks (see `docs/deployment.md`).

## Documents
`docs/runbook_local.md` · `docs/false_positive_investigation.md` · `docs/deployment.md` · `docs/prototype_plan.md` · `docs/model_investigation.md` · `docs/evaluation_plan.md` · `docs/report_outline.md` · `ATTRIBUTIONS.md`.

## Remaining limitations
Face detection is the weak OpenCV Haar frontal cascade; the visual model's training data is undisclosed; AASIST is out-of-domain for most real-world audio; frame timestamps are nominal; no calibration, no evaluation, no synchronisation analysis; PDF export is the browser's print dialog; one analysis runs at a time; no authentication; nothing has been deployed or load-tested.
