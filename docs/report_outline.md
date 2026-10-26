# Report outline (honest-scope version)

Every sentence in the report must fit exactly one of these four categories. Label them in the text.

| Category | Meaning | Examples in this project |
|---|---|---|
| **Implemented** | Code exists in the repository | preprocessing, branches, fusion rule, API, UI, privacy controls |
| **Pretrained-model inference** | Third-party weights run as published; nothing trained here | AASIST (audio), Hugging Face ViT (visual) |
| **Software testing** | Tests of code behavior with synthetic fixtures or test doubles | pytest, vitest, API integration tests, mutation checks |
| **Future model evaluation** | Measuring detection performance on labelled held-out data | NOT done. Listed as future work only |

## Suggested structure
1. **Abstract and scope.** A working prototype of a multimodal analysis pipeline using pretrained detectors; state in one sentence that no training or performance evaluation was performed.
2. **Background and related work.** Deepfake detection, audio anti-spoofing (ASVspoof, AASIST), face-forgery detection, multimodal fusion. Cite original papers only after checking them (AASIST: Jung et al., ICASSP 2022; ViT: Dosovitskiy et al., ICLR 2021; FakeAVCeleb: Khalid et al. 2021, as *planned* dataset).
3. **System design (Implemented).** Architecture diagram: upload -> validation (ffprobe) -> job -> audio branch / visual branch -> score-level fusion -> result store -> UI. Describe the modality-status contract (available, missing, corrupted, unreliable, not_analyzed) and why absence is never treated as authenticity.
4. **Components.**
   - Audio branch: FFmpeg decoding to 16 kHz mono, energy gate, 4.04 s windows, AASIST (*pretrained inference*), window scores with timestamps.
   - Visual branch: uniform frame sampling, Haar face detection (largest face), ViT classifier on crops (*pretrained inference*), frame scores with nominal timestamps.
   - Fusion A: equal-weight mean over usable modalities, default threshold 0.5, both explicitly unvalidated; no learned fusion.
   - Backend: job states, limits, safe temp files, cleanup, structured errors. Frontend: real upload progress, real stage polling.
5. **Model attribution and limitations.** From `ATTRIBUTIONS.md` and the model cards. Include: AASIST trained on ASVspoof 2019 LA; a third-party benchmark reports near-chance error on in-the-wild audio (cite as reported, not reproduced); ViT training data undisclosed.
6. **Privacy and responsible use.** No raw-video retention, no file names stored, results deletable, no forensic claims.
7. **Software testing (Software testing).** Counts and commands taken from saved test output; mutation checks; what test doubles are and why their outputs are not results. State plainly that passing tests does not show detection accuracy.
8. **Demonstration (Pretrained-model inference).** Screenshots and the saved `result.json` of your own consented clip. Describe what the output shows, not whether the clip is "correct". A single clip is a demo, not evidence of performance.
9. **Limitations.** Uncalibrated scores; unvalidated threshold/weights; domain mismatch; Haar detector weakness; no temporal or synchronization modeling; speech-only audio model; 60 s limit; single-worker queue; no authentication.
10. **Future work (Future model evaluation).** Obtain FakeAVCeleb (pending) and/or ASVspoof 2019 LA; identity-grouped splits; thresholds and calibration chosen on validation data; macro-F1, per-class recall, ROC-AUC, EER with grouped-bootstrap confidence intervals; robustness (compression, noise, missing modality); training/fine-tuning; reliability-aware fusion; synchronization analysis. The plan is in `docs/evaluation_plan.md`.
11. **References.** Verified against original sources.

## Claims you must NOT make
- Any accuracy, precision, recall, F1, ROC-AUC, EER, or "detects X% of deepfakes".
- That the system "works", "is validated", or "is reliable" at detecting deepfakes.
- That a low score means the media is real, or that a missing audio/video track is a sign of anything.
- That FakeAVCeleb or any other dataset was used.
- That the pretrained models were trained, fine-tuned or evaluated by you.

## Evidence table (fill from saved files, not memory)
| Claim in report | Evidence file/command |
|---|---|
| N Python tests pass | `pytest_output.txt` |
| N UI tests pass; build succeeds | `cd frontend; npm test; npm run build` output |
| AASIST/ViT load on the author's machine | outputs of `probe_aasist.py`, `probe_video_model.py` |
| Demo clip result | `result.json` + screenshots |
