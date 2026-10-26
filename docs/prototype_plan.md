# Deadline-focused prototype plan (submission: 27 Oct 2026)

> **Status update (25 Oct):** the backend and frontend described below are now implemented and tested; see `README.md` for the current implemented/verified table and `docs/runbook_local.md` for commands. The scope, honesty rules, and data policy in this file still apply.

Supersedes the phase order in `decision_log.md`; Phase 0 work and tests are preserved.
Written 24 Oct 2026. FakeAVCeleb access is pending and **nothing depends on it**.

## What the prototype is (and is not)
- **Is:** a working local application that runs *pretrained* detectors on an uploaded video: an audio branch (AASIST) and a visual branch (a Hugging Face ViT face classifier), with preprocessing, per-branch results with timestamps, a documented untrained score-level fusion, explicit handling of missing/corrupted/unreliable modalities, and honest labelling.
- **Is not:** a trained system, a validated detector, or an evaluated one. **No training is performed. No accuracy/F1/AUC is claimed.** Evaluation is future work unless the optional step below is completed and its outputs are saved.

## Terminology used everywhere (code, UI, report)
| Term | Meaning here |
|---|---|
| Pretrained-model inference | Running third-party weights as published. This is all the prototype does. |
| Training | Not done. (Future work: train on FakeAVCeleb/ASVspoof once access and time allow.) |
| Evaluation | Measuring performance on labelled held-out data with saved outputs. Not done yet. Software tests on synthetic fixtures are **not** evaluation. |

## Scope with cut lines
**MUST (demo + honesty):** audio branch verified; visual branch verified on your machine; CLI `analyze_video.py`; FastAPI backend (job states queued/processing/completed/failed, upload validation, no raw-video retention, deletion); simple polished frontend (upload, real progress, separate audio/visual results, fusion, errors, model/limitation panel, "uncalibrated" labelling); tests; README + runbook; report with limitations.
**SHOULD:** prediction history (results only, SQLite); JSON report download; frame/segment timeline; DeepfakeTIMIT smoke run (clearly a smoke run, not evaluation).
**COULD (only if time remains):** YuNet face detector; AASIST vs AASIST-L comparison; ASVspoof 2019 LA in-domain reproduction; HTML/PDF report.
**WON'T before 27 Oct:** synchronization model, training, reliability-aware feature fusion, GPU/Docker deployment, FakeAVCeleb/FF++ experiments.

## Timeline
| Date | Goal | Exit criterion |
|---|---|---|
| Sat 24 Oct (today) | Model investigation; audio branch built and tested here; probes for your machine | Probe scripts delivered; **you run A1-A4 in `runbook_local.md` and report output** |
| Sun 25 Oct | Fix whatever your probes reveal; FastAPI backend with real inference path; backend integration tests | `uvicorn` serves; upload -> result via curl using your real models |
| Mon 26 Oct | Frontend; history/report/deletion; end-to-end run on your own consented clip; smoke run on permitted sample if available | Demo works from a clean start; screenshots captured |
| Tue 27 Oct | Freeze: clean-clone verification, README/report/limitations, submission package | Checklist in `runbook_local.md` section F passes |

**Decision point, Mon 26 Oct noon:** if the React toolchain causes trouble, ship a single-page HTML/JS frontend served by FastAPI (same API, same states).

## Data and sample policy (use only what permits the intended use: academic, non-commercial project)
| Source | Use | Status |
|---|---|---|
| Your own recordings with the consent of everyone shown/heard | Pipeline demo; real-media smoke runs | Allowed (you hold the rights). Not an evaluation set. |
| ASVspoof 2019 LA (ODC-By, Edinburgh DataShare) | Future audio evaluation; in-domain reproduction of AASIST | License verified; 7.1 GB download; optional |
| DeepfakeTIMIT (Zenodo 4068245, "Open"; 226.6 MB; GAN face-swaps of VidTIMIT, original audio kept) + VidTIMIT (research use permitted, citation required) | Optional smoke run | Zenodo licence text **not verified** by me: read the record page before use. Old low-quality face swaps; may not resemble what the ViT model was trained on. |
| FakeAVCeleb | Future work | Pending; do not rely on it |
| FaceForensics++ | Future work | Not verified; do not rely on it |
| Synthetic fixtures (ffmpeg patterns, public-domain NASA photo) | Software tests only | Never reported as performance |

## Honesty rules enforced in code
1. Scores are called *raw model scores*; `calibrated: false` and `validated_on_project_data: false` appear in every result and must be shown in the UI.
2. A missing, corrupted, unreliable, or not-analysed modality never contributes a score; with no usable modality the verdict is `insufficient_evidence`.
3. "Not flagged" is never phrased as "authentic".
4. Every result carries model name, version/commit, license, citation, training-data note and limitations. Test-fixture models set `is_test_fixture`, and the UI must show a banner.
5. No file paths in results, no raw video retained after analysis.
