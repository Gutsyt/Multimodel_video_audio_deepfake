# Local runbook (Windows PowerShell first; macOS/Linux notes inline)

Run every command from the repository root (the folder containing `pyproject.toml`) unless a step says otherwise.
Anything marked **[untested on your machine]** was written but has not been run by me on Windows. Paste errors to me verbatim.

## A. Prerequisites
| Need | Check | Install |
|---|---|---|
| Python 3.10, 3.11 or 3.12 (I tested 3.12 only) | `python --version` | python.org, tick "Add to PATH" |
| FFmpeg + ffprobe on PATH | `ffmpeg -version` | `winget install Gyan.FFmpeg`, then **open a new PowerShell** |
| Node.js 20.19+ or 22.12+ (ONLY to rebuild/test the UI; a prebuilt UI is included) | `node --version` | `winget install OpenJS.NodeJS.LTS` |

## B. One-time setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass   (macOS/Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU build; if unreachable use: pip install torch
pip install -e ".[video,hf,api]"
pip install -r requirements-dev.txt
```

## C. Environment check and the two model probes (DO THIS FIRST; send me the outputs)
I have **not** claimed that either model works on your machine. These commands decide that.
```powershell
python scripts/check_environment.py --json | Out-File -Encoding utf8 env_report.json; Get-Content env_report.json
python -m pytest -q 2>&1 | Tee-Object pytest_output.txt
python scripts/probe_aasist.py --variant AASIST
python scripts/probe_aasist.py --variant AASIST --media .\my_clip.mp4
python scripts/probe_video_model.py --image .\face.jpg
python scripts/probe_video_model.py --video .\my_clip.mp4
# Only if the pinned Hugging Face revision is rejected:
python scripts/probe_video_model.py --image .\face.jpg --revision main
python scripts/analyze_video.py .\my_clip.mp4 --out result.json
```
Pass criteria: `probe_aasist.py` prints `ALL CHECKS PASSED`; `probe_video_model.py` prints `PASS load` with `id2label`/`fake_index`, then `ALL CHECKS PASSED`.
The first visual probe downloads about 350 MB. `my_clip.mp4` is a 10-20 s clip of yourself (everyone shown/heard must consent); `face.jpg` is any face photo you have the right to use.
Send me: `env_report.json`, the last lines of `pytest_output.txt`, and the full output of the four probe commands plus `result.json`.

## D. Run the application
**Mode 1: one terminal, prebuilt UI (recommended for the demo)**
```powershell
python scripts/run_server.py
```
Open http://127.0.0.1:8000. The header chips show each model's real state (`loading` while the visual model downloads on first run).
If the visual model cannot load, run audio-only: `$env:AVDD_VIDEO_MODEL="none"; python scripts/run_server.py` (the UI will report the visual branch as not analyzed; it never shows that as authentic).

Useful settings (PowerShell `$env:NAME="value"` before starting; unset with `Remove-Item Env:NAME`):
`AVDD_AUDIO_MODEL` (`AASIST`, `AASIST-L`, `none`), `AVDD_VIDEO_MODEL` (`hf`, `none`), `AVDD_MAX_UPLOAD_MB` (100), `AVDD_MAX_DURATION_S` (60), `AVDD_DATA_DIR` (`app_data`), `AVDD_HF_REVISION`, `AVDD_DEVICE` (`cpu`), `AVDD_ENABLE_DOCS` (`0` hides /docs).

**Mode 2: development, two terminals**
```powershell
# Terminal 1 (repo root, venv active)
python scripts/run_server.py
# Terminal 2
cd frontend
npm ci
npm run dev        # http://localhost:5173 ; /api is proxied to port 8000
```
Rebuild the shipped UI after editing it: `cd frontend; npm ci; npm run build` (output in `frontend/dist`).

## E. Using the API without the UI
```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/capabilities | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/analyses -InFile .\my_clip.mp4 -ContentType "application/octet-stream"
Invoke-RestMethod "http://127.0.0.1:8000/api/analyses/$($r.id)" | ConvertTo-Json -Depth 8
```
(curl: `curl.exe -X POST --data-binary "@my_clip.mp4" -H "Content-Type: application/octet-stream" http://127.0.0.1:8000/api/analyses`.) The upload is the raw file body; the file name is never read or stored. Interactive docs: http://127.0.0.1:8000/docs.
Endpoints: `GET /api/health`, `/api/ready`, `/api/capabilities`; `POST /api/analyses` (202); `GET /api/analyses`, `/api/analyses/{id}`, `/api/analyses/{id}/report`; `DELETE /api/analyses/{id}`. Errors are `{"error": {"code", "message", "details"?}}`.

## F. Tests, lint, types
```powershell
python -m pytest -q
ruff check .
mypy
cd frontend; npm ci; npm run typecheck; npm test; npm run build; cd ..
```
Optional real-HTTP UI-client test against your running server (set CORS for the jsdom origin before starting the server):
```powershell
# Terminal 1:  $env:AVDD_CORS_ORIGINS="http://localhost:3000"; python scripts/run_server.py
# Terminal 2:
cd frontend
$env:AVDD_E2E_URL="http://127.0.0.1:8000"; $env:VITE_API_BASE="http://127.0.0.1:8000"
$env:AVDD_E2E_VIDEO="C:\full\path\my_clip.mp4"; $env:AVDD_E2E_EXPECT="real"
npx vitest run src/e2e.test.ts
```
(`AVDD_E2E_EXPECT=real` requires the audio model to be loaded.)

## G. Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: avdd` | venv not active or wrong folder; `pip install -e ".[video,hf,api]"` from the repo root |
| `ffmpeg`/`ffprobe` not found | install FFmpeg; open a NEW terminal |
| Chip says `visual: failed (download_failed)` | internet/proxy; try `$env:AVDD_HF_REVISION="main"`; `pip install -U transformers` |
| Chip says `visual: failed (ambiguous_labels)` | send me the printed labels |
| Upload returns `503 models_loading` | wait for the chips to show `loaded` |
| `422 duration_exceeded` / `413 file_too_large` | clip is over the limits; trim it or raise `AVDD_MAX_DURATION_S` / `AVDD_MAX_UPLOAD_MB` |
| Port 8000 busy | `python scripts/run_server.py --port 8001` and open that port |
| Page is blank | check `frontend\dist\index.html` exists (else `cd frontend; npm ci; npm run build`); open browser DevTools console |
| `npm ci` engine warning | upgrade Node (A) |
| Log: "could not delete a temporary upload" | harmless: a Windows file lock; the file is removed at next server start |
| Audio score near 1.0 for music/noise/tones | expected: the model is for speech; see its limitations in the UI |

## H. What is and is not verified
| Item | Status |
|---|---|
| Backend: integration tests with test doubles, plus real-model smoke tests in `tests/real` | **Run here** (Linux, Python 3.12) |
| Frontend: type check, 38 UI tests, production build, real-HTTP client test vs. doubles server and vs. real-AASIST server | **Run here** (Node 22) |
| `curl` upload to a real server with real AASIST weights; structured errors; no leftover temp files; restart sweep | **Run here** |
| Any command on Windows PowerShell; Python 3.10/3.11; Node 20 | **Not run** |
| Hugging Face ViT download and inference (visual branch with a real model) | **Not run** (unreachable from my sandbox): needs your probe |
| How the UI looks and behaves in a real browser (layout, drag-and-drop, video preview) | **Not run**: I have no browser; DOM tests and a served build only. Please look and send screenshots |
| Detection quality of either model on real deepfakes | **Not evaluated** (future work) |

## I. Final checklist (27 Oct)
- [ ] Section C passes on your machine and I have seen the outputs
- [ ] `python scripts/run_server.py` -> upload your clip in the browser -> separate audio/visual cards, fused card, timeline, report download, delete
- [ ] Screenshots: idle, in-progress, a result with a missing/unreliable modality, an error
- [ ] Report numbers come only from saved outputs (`pytest_output.txt`, `result.json`); no accuracy/F1/AUC claimed
- [ ] No videos, `.env`, or `app_data/` in the submission


## Optional modalities, audio-only and image-only uploads (added 26 Oct 2026)
- Accepted by content (not by extension): video (MP4, MOV, MKV, WebM, AVI), audio (WAV, MP3, FLAC, OGG, M4A, AAC) and images (JPEG, PNG, WebP, BMP). Animated GIF and TIFF are rejected.
- Each analysis reports per branch whether it ran, and why not: no audio track, silent audio, no face found, corrupted stream. With one usable branch the result is that branch's own score and decision, with no fused score. With none the result is `insufficient_evidence`.
- Inspect a file branch by branch: `python scripts/diagnose_media.py .\clip.mp4 --sensitivity` (see `docs/false_positive_investigation.md`).
- Tests: mocked-detector tests are in `tests/unit` and `tests/integration`; `tests/real` runs the real models (plumbing only) and is skipped when torch or the Hugging Face download is unavailable. Deselect with `-m "not real_models"`.
- Frontend: `frontend/.env.example` documents `VITE_API_BASE`. Deployment notes: `docs/deployment.md`.
