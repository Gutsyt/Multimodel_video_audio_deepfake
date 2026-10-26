# AVDD user guide: run it locally and test every feature

This guide takes you from a clean Windows machine to a running app, then walks through each feature with files you can create yourself. Commands are Windows PowerShell; macOS/Linux notes are at the end. Scores are **raw model outputs, not probabilities**, and "flagged" only means a score crossed an unvalidated default threshold.

## 1. Install
You need Python 3.10+, Node 20+ and ffmpeg (which includes ffprobe).
```powershell
winget install Gyan.FFmpeg          # then close and reopen PowerShell
ffmpeg -version; ffprobe -version   # both must print a version
python --version; node --version
```

## 2. Set up (once)
Run from the repository root.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[video,hf,api]"
pip install -r requirements-dev.txt
cd frontend; npm ci; npm run build; cd ..
```

## 3. Check the models load
```powershell
python scripts/probe_aasist.py
python scripts/probe_video_model.py --image .\face.jpg
```
The second command downloads the visual model (about 350 MB) from Hugging Face the first time. It should print the labels `{0: 'Realism', 1: 'Deepfake'}`.

## 4. Start the app
```powershell
python scripts/run_server.py
```
Open **http://127.0.0.1:8000**. The first minute the sidebar shows the models as `loading`; wait until both say `loaded`. Stop with Ctrl+C.

Optional hot-reload UI for development: keep the backend running, then in a second terminal `cd frontend; npm run dev` and open http://localhost:5173.

## 5. Create test files
These are synthetic or public files, useful for checking behaviour, **not** for judging detection quality. For a real face photo use `face.jpg` from the repo; for real speech use any WAV/MP3/MP4 of your own.
```powershell
mkdir testmedia; cd testmedia
Copy-Item ..\face.jpg .\face.jpg
# audio-only: a 6 s tone (a tone is not speech, so the audio branch may call it unreliable or give an arbitrary score)
ffmpeg -y -f lavfi -i "sine=frequency=300:duration=6:sample_rate=16000" tone.wav
# video with a face and audio (still photo looped as video + tone)
ffmpeg -y -loop 1 -framerate 10 -i face.jpg -f lavfi -i "sine=frequency=300:duration=6:sample_rate=16000" -t 6 -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:a aac -shortest face_audio.mp4
# video with a face, no audio track
ffmpeg -y -loop 1 -framerate 10 -i face.jpg -t 6 -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" face_noaudio.mp4
# video with audio but no face
ffmpeg -y -f lavfi -i "testsrc=duration=6:size=320x240:rate=10" -f lavfi -i "sine=frequency=300:duration=6:sample_rate=16000" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest noface_audio.mp4
# video with neither usable audio nor a face
ffmpeg -y -f lavfi -i "testsrc=duration=3:size=320x240:rate=10" -c:v libx264 -pix_fmt yuv420p neither.mp4
# bad files
"this is not media" | Out-File notes.txt
New-Item empty.wav -ItemType File
[IO.File]::WriteAllBytes("corrupted.mp4", (Get-Content face_audio.mp4 -Encoding Byte -TotalCount 5000))
cd ..
```

## 6. Feature walkthrough
Use **New analysis** in the sidebar. Tick the consent box, then **Run analysis**. The progress list shows the real upload bytes and the real backend stages.

| Try this file | You should see |
|---|---|
| `face_audio.mp4` | Both branches ran. A **Fusion** panel shows a fused score, the weights (0.5 / 0.5), the rule, and "unvalidated" labels. Basis: "Audio and visual branches, fused". |
| `face_noaudio.mp4` | Audio branch "Not in this file: no audio track". Result is **visual only**, labelled a single-branch result. Fusion says "not performed" and there is **no fused score**. |
| `noface_audio.mp4` | Visual branch "Too unreliable to score: faces found in 0 of 16 frames". Result is **audio only**, no fused score. |
| `neither.mp4` | **Insufficient evidence**. No scores anywhere, and the page states this is not evidence the media is authentic. |
| `tone.wav` (or any WAV/MP3/FLAC/OGG/M4A) | Audio-only upload; the visual branch says "this audio file has no video stream". |
| `face.jpg` (or PNG/WebP/BMP) | Image-only upload; the audio branch says "this image has no audio", and the visual branch shows one score and a face-position box. If the image has no detectable face you get **Insufficient evidence**. |
| `notes.txt` | Rejected in the browser before upload: "Unsupported file type". |
| `empty.wav`, `corrupted.mp4` | Rejected with a clear message ("The file is empty", "could not be read; it may be corrupted"). Nothing is stored. |
| A file over the size limit | Rejected with the limit shown (default 100 MB; audio/video up to 60 s; images up to 40 MP). |

On a result page, check:
- The banner reads **Flagged**, **Not flagged** or **Insufficient evidence**, always with the note that "not flagged" does not mean authentic.
- Each branch card shows the raw score on a 0 to 1 meter with the threshold tick, the audio window bars or visual frame dots over time (frames with no face are crosses on their own row, never zeros), and expandable diagnostics: what the score means, statistics, preprocessing, per-window/per-frame logits and the model's limitations.
- A warning appears when windows or frames disagree strongly; the mean hides that.
- **Download JSON** saves the stored result exactly as returned. **Print / save as PDF** opens the browser print dialog with a light print layout. **Delete** removes the record.

## 7. Other pages
- **Dashboard**: stored-run squares, outcome bar, raw scores by run, which branches ran, and live model health. On a fresh install it shows an empty state and no invented numbers.
- **History**: search, filter by outcome and media kind, sort, open, per-record JSON export, delete, and "Delete all finished". It states exactly what is stored (results only, never the media or its file name).
- **Reports**: completed analyses with JSON download and open-to-print.
- **System**: live model state and model cards, limits, API address, privacy text, and a **Re-check** button that calls `/api/ready`. Stop the backend and reload to see the "backend unreachable" state.

## 8. Understand one file in depth
```powershell
python scripts/diagnose_media.py .\testmedia\face_audio.mp4
python scripts/diagnose_media.py .\testmedia\face_audio.mp4 --sensitivity --json report.json
```
This prints per-window and per-frame logits, input-quality metrics, and (with `--sensitivity`) how much the score moves under harmless preprocessing changes. Read `docs/false_positive_investigation.md` before drawing conclusions from any single file.

## 9. Run the automated tests
```powershell
python -m pytest -q                       # backend (stand-in detectors + real-model smoke tests)
python -m pytest -q -m "not real_models"  # skip the real-model tests
ruff check .
mypy
cd frontend; npm run typecheck; npm test; npm run build
```
Most tests use stand-in detectors with fixed scores and synthetic media: they prove behaviour, not accuracy. `tests/real` runs the real models for plumbing only and skips itself if they are unavailable. Known pre-existing items: `tests/unit/test_inventory.py` is skipped (its module was never committed); ruff reports shebang file-mode warnings; mypy reports two errors.

## 10. Troubleshooting
| Symptom | Fix |
|---|---|
| "ffprobe is not installed" / `tool_missing` | Install ffmpeg and reopen PowerShell so it is on PATH. |
| Models stay `loading` or visual shows `failed (download_failed)` | Check internet access to huggingface.co, then restart the server. |
| "Backend unreachable" in the UI | The server is not running, or you opened the dev UI without the backend on port 8000. |
| Upload rejected as too large or too long | Raise `$env:AVDD_MAX_UPLOAD_MB` / `$env:AVDD_MAX_DURATION_S` before starting the server. |
| Port 8000 in use | `python scripts/run_server.py --port 8001`. |
| Blank page after code changes | Rebuild: `cd frontend; npm run build`. |

Settings are environment variables (`AVDD_*`, see `.env.example`). History is stored in `app_data/` and can be deleted from the History page or by removing that folder while the server is stopped.

macOS/Linux: use `python3`, `source .venv/bin/activate`, `brew install ffmpeg` / `apt install ffmpeg`, and the same `ffmpeg` and `npm` commands (drop the PowerShell-specific `Out-File`, `Get-Content` and `New-Item` lines: use `echo "x" > notes.txt`, `: > empty.wav`, `head -c 5000 face_audio.mp4 > corrupted.mp4`).

## Limits to keep in mind
Pretrained models, no calibration, no accuracy evaluation; Haar face detection is weak; the server has no authentication, so keep it on your own machine or a private network. Nothing here is forensic or legal proof.
