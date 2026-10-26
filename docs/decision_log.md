# Decision log

| # | Decision | Reason | Evidence | Revisit when |
|---|---|---|---|---|
| D1 | FakeAVCeleb is the primary dataset candidate | Only candidate with per-modality labels plus both streams | Official README + paper | Inventory contradicts expectations (H1-H10) |
| D2 | No dataset-dependent code (label mapping, splits, loaders) before the real inventory | Official sources do not describe directory/metadata layout; only a third-party loader hints at it | Phase 0 research | Inventory delivered |
| D3 | Own identity-grouped splits | No official split in README; papers use their own | README; arXiv 2311.02733, 2605.19630 | Authors ship splits with the data |
| D4 | Baselines stay EfficientNet-B0 (visual) and log-Mel CNN (audio); Fusion A before Fusion B | User directive | – | – |
| D5 | Sync is optional and gated (G1-G4) | Weights license second-hand; no weights obtained; sync labels do not exist in the dataset | `phase0_findings.md` section 3 | Gates pass |
| D6 | Do not use `facenet-pytorch` | Pins `torch<2.3`, `numpy<2` | PyPI metadata for 2.6.0 | A compatible release appears |
| D7 | Phase 0 code uses only the standard library | Keeps tooling testable here; ML deps need a GPU-capable or large-disk environment | – | Phase 2 |
| D8 | Modality status contract: missing/corrupted modalities cannot carry scores | Prevents "missing == real" bugs structurally | `types.py`, contract tests, mutation check | – |
| D9 | Local CLI inference before API/frontend | User directive | – | – |
| D10 | Pins come from versions actually installed and tested | Package versions on PyPI are newer than my background knowledge | `requirements-dev.txt` | Each phase adds deps only after install + test |

## Revised phase order
0 Foundations (this) -> 1 Manifest & splits -> 2 Video pipeline + EfficientNet-B0 -> 3 Audio pipeline + log-Mel CNN -> 4 Fusion A + thresholds + calibration + missing-evidence tests -> **5 Local end-to-end inference CLI with real checkpoints** -> 6 Fusion B + ablations + robustness suite -> 7 Sync (only if gates pass; can slot in earlier) -> 8 FastAPI backend -> 9 React frontend -> 10 History/reports/privacy -> 11 QA/Docker/docs -> 12 Paper artifacts.

## Deadline pivot (24 Oct 2026)
| # | Decision | Reason |
|---|---|---|
| D11 | Pretrained inference only; no training; evaluation is future work | No lawful labelled data available before the deadline; honesty |
| D12 | Audio: vendored AASIST (MIT), default `AASIST`, `AASIST-L` selectable | Verified loadable here; tiny; pure PyTorch |
| D13 | Visual: Hugging Face ViT face classifier, label index read from `config.id2label` (never assumed) | Only licence-clear, light candidate found; unverified until user's probe |
| D14 | Haar face detector (bundled) now, YuNet later | YuNet file is in Git LFS and untestable here |
| D15 | `opencv-python-headless<5` | YuNet/OpenCV 5.x model-file incompatibility noted upstream |
| D16 | New status `NOT_ANALYZED` (model missing/failed) | A system limitation must not look like a media property or like authenticity |
| D17 | Fusion A = equal-weight mean over usable modalities, default threshold 0.5, both flagged unvalidated | No validation data exists to fit anything |
| D18 | No temp files: audio decoded via FFmpeg pipe; results carry no paths | Privacy |
