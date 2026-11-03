# Testing: feature checklist

Start the app: `python scripts/run_server.py`, then open http://127.0.0.1:8000. Wait until the header chip says **Models ready**.

Test files you can make yourself:
```bash
ffmpeg -f lavfi -i sine=frequency=300:duration=5 tone.wav                         # audio, no speech
ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=25 -pix_fmt yuv420p noface.mp4   # video, no face
echo "not media" > notes.txt
head -c 20000 some_real_video.mp4 > truncated.mp4                                 # broken file
```
Plus: one genuine video of a person talking (your phone), one image of a face, one known deepfake (for example from `$E2E/A_ff/fake`).

### Automated checks (run first)
- [ ] `python -m pytest -q -m "not real_models"`: all pass
- [ ] `python -m pytest -q -m real_models`: passes once the heads exist (loads real backbones; slow the first time)
- [ ] `ruff check .` and `mypy`: clean
- [ ] `cd frontend && npm run typecheck && npm test && npm run build`: clean

### Backend / API
- [ ] `curl http://127.0.0.1:8000/api/health` returns `{"status":"ok"}`
- [ ] `curl http://127.0.0.1:8000/api/ready` returns 503 while loading, then 200
- [ ] `curl http://127.0.0.1:8000/api/capabilities` shows both models `loaded`, each card with `metrics`
- [ ] Upload by API: `curl -X POST --data-binary @clip.mp4 -H "Content-Type: application/octet-stream" http://127.0.0.1:8000/api/analyses` returns an `id`; `GET /api/analyses/<id>` eventually shows `completed` with `assessment.verdict`
- [ ] `GET /api/analyses/<id>/report` downloads JSON with no file name or path inside
- [ ] After an analysis, `app_data/uploads_tmp/` is empty (the upload was deleted)

### Analyze page
- [ ] Drag a file onto the drop zone: border turns solid and it scales up; drop selects the file
- [ ] Click the drop zone: the file picker opens; keyboard Enter/Space also opens it
- [ ] Preview shows the video/image/audio player
- [ ] `notes.txt` is rejected with a readable message before upload
- [ ] A file over the size limit is rejected with the MB limit shown
- [ ] **Analyze**: the steps (Uploading → Reading → Voice → Face → Weighing) tick through, the scan line sweeps the preview, and the page opens the result
- [ ] Cancel during upload works and shows "Upload cancelled"
- [ ] With the backend stopped: the chip says "Backend offline" and the page shows the warning

### Verdicts (each must appear correctly)
- [ ] Genuine talking-head video gives **Likely authentic** (or Inconclusive), and not Likely manipulated
- [ ] Known deepfake gives **Likely manipulated** (or Inconclusive)
- [ ] `noface.mp4` gives **Not enough to judge**; the Face row reads "Not enough signal", with no score shown
- [ ] `tone.wav` gives **Not enough to judge**; the voice is unusable (no speech)
- [ ] Face image gives a face-only result; the Voice row reads "Not in this file"
- [ ] Speech-only audio file gives a voice-only result with a waveform
- [ ] `truncated.mp4` gives a clear "could not be read" error, not a verdict

### Result page
- [ ] The verdict word comes into focus after the scale markers settle; with reduced motion turned on in the OS it appears immediately
- [ ] Evidence scale: each check's marker lands in the right band (genuine / borderline / manipulated) and that band lights up
- [ ] Replay (same tab only): the video plays and the face box follows the face, coloured by that frame's score
- [ ] Filmstrip: crops appear; clicking one jumps the video to that frame; hovering shows the time
- [ ] Timeline: hovering shows the crosshair and the time / Face % / Voice % readout; clicking seeks; arrow keys step 1 s
- [ ] Waveform drawn behind the voice windows (same tab only)
- [ ] Reload the page: replay, filmstrip and waveform disappear and the hint explains why; scale and timelines remain
- [ ] **Compare**: pick an earlier analysis; both dots and the point difference per check appear
- [ ] **Share card**: downloads a PNG (desktop) or opens the share sheet (phone); no face in the image
- [ ] **Copy summary**: the clipboard contains the verdict and summary
- [ ] **JSON** downloads the report; **PDF** opens print with the header and buttons hidden
- [ ] **Delete** removes the record and goes to History
- [ ] Open a result while it is still analysing: live steps and a skeleton show, then the result appears

### History / Models / general
- [ ] History lists runs; the filters (All / Authentic / Inconclusive / Manipulated) work; delete one; Clear all asks for confirmation
- [ ] Models page shows each head's real AUC, EER, genuine-flagged and fakes-caught per test set, and the thresholds
- [ ] Phone width (390 px, or DevTools device mode): no horizontal scroll; nav, scale, timelines and filmstrip usable
- [ ] Keyboard only: Tab reaches every control with a visible focus ring
- [ ] An old record from the previous version opens with "made by an older model version" rather than crashing

---
