# avdd: multimodal audio-visual deepfake analysis (research prototype)

**Status (25 Oct 2026): local prototype with a working backend and frontend. Pretrained-model INFERENCE only: no training, no performance evaluation, no accuracy claims.**
Scores are raw model outputs: uncalibrated and not validated on any dataset by this project. Output is not forensic or legal proof.

## What is implemented, tested, and not yet verified
| Capability | Implemented | Software-tested (synthetic fixtures / test doubles) | Verified with real pretrained weights |
|---|---|---|---|
| ffprobe parsing, modality-status contract (missing/corrupted/unreliable/not_analyzed are never "authentic") | yes | yes | n/a |
| Audio branch: FFmpeg decode, energy gate, 4 s windows, AASIST | yes | yes | **AASIST/AASIST-L run in the author's Linux sandbox**; not yet on the user's machine |
| Visual branch: frame sampling, Haar face detection, Hugging Face ViT on face crops | yes | yes (stub detectors) | **No**: model download/run pending the user's probe |
| Score-level fusion (equal-weight, unvalidated default threshold) | yes | yes | n/a |
| FastAPI backend: validated uploads, size/duration limits, job states, temp-file cleanup, history, JSON report, delete | yes | yes (25 integration tests) | one real-AASIST test + curl run |
| React frontend: upload with real progress, real stage polling, separate audio/visual cards, fused card, timeline, errors, model limitations, history | yes | yes (22 UI tests, type check, build) | real-HTTP client test vs. a real-AASIST server; **not viewed in a real browser by the author** |
| Training, evaluation (accuracy/F1/AUC), calibration, synchronization, reliability-aware fusion, Docker | **no** | no | no (future work) |

## Quick start (Windows PowerShell; details and macOS/Linux in `docs/runbook_local.md`)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[video,hf,api]"; pip install -r requirements-dev.txt
python scripts/probe_aasist.py; python scripts/probe_video_model.py --image .\face.jpg   # verify models first
python scripts/run_server.py                                                           # then open http://127.0.0.1:8000
```

## Documents
`docs/runbook_local.md` (commands) · `docs/prototype_plan.md` (scope) · `docs/model_investigation.md` · `docs/report_outline.md` · `docs/evaluation_plan.md` (future evaluation) · `ATTRIBUTIONS.md` · `docs/phase0_findings.md` (dataset research; FakeAVCeleb not required).

## Architecture
`src/avdd/{media,audio,video,fusion,pipeline,api,common}`; `third_party/aasist` (vendored, MIT); `frontend/` (React + Vite + TypeScript; `frontend/dist` is prebuilt); `scripts/`; `tests/`.
Upload (raw body) -> ffprobe validation -> queued job -> `analyze_media` (audio branch, visual branch, fusion) -> result in SQLite (no media, no file name) -> UI.

## Privacy and retention
The uploaded video exists only as a randomly named temporary file while its job runs and is deleted when the job ends (success or failure); leftovers are swept at startup. Only the result JSON is stored, locally, until you delete it. Nothing is sent to third parties except the one-time model download from Hugging Face.

## Responsible use
Genuine media can be flagged and manipulated media can evade detection. "Not flagged" is never evidence of authenticity.
