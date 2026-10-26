# Evaluation plan (refined after Phase 0 review)

Principles: validation data selects models, thresholds, and calibration; the test set is touched once at the end; every reported number is generated from saved experiment outputs; software-test fixtures never produce reported performance.

## Metrics (all reported with sample sizes)
- Accuracy, precision, recall, **macro F1**, **per-class recall** (real recall and fake recall separately), ROC-AUC, confusion matrix.
- **EER** for audio anti-spoofing experiments (plus ASVspoof protocol results where used).
- Inference time and memory (measured, with hardware stated).

## Thresholds
Chosen on the validation set only (rule fixed in the config before running, e.g. maximize validation macro F1, or fixed false-positive rate on real). The frozen threshold is then applied unchanged to test. Threshold, rule, and validation size are saved with the checkpoint.

## Calibration
- Fit calibrator (temperature scaling / Platt / isotonic: compared on validation by Brier score and log-loss) on validation only.
- Assess on held-out data with Expected Calibration Error (fixed binning, reported with the bin count), Brier score, and reliability diagrams.
- Only if calibration is acceptable by a criterion written down *before* the test run does the UI/report show a "calibrated probability". Otherwise it shows a raw model score, labeled as such. With few real videos, calibration estimates will be noisy: report that.

## Uncertainty
- 95% confidence intervals by **cluster bootstrap over identity groups** (videos of one identity are correlated; naive per-video bootstrap would be overconfident). Number of resamples recorded.
- Multiple random seeds (target >= 3) for trained components where compute allows; report mean and spread, or state that only one seed was feasible.
- Real-class counts are small in FakeAVCeleb (about 500 real videos in total per the official README; the test share will be smaller still): intervals will be wide and must be shown.

## Experiments
Video-only; audio-only; Fusion A (score-level; weights from validation) **before** Fusion B (reliability-aware feature fusion); ablations (each fusion component); compression/resolution; noise/audio codec; missing or corrupted modality; unseen identity / method (leave-one-method-out); silence-trimming and codec-uniformity shortcut checks; sync only if gates in `phase0_findings.md` pass.

## Missing / corrupted / unreliable evidence: required tests
Contract level (done in Phase 0: `tests/unit/test_modality_contract.py`). Required again at each layer:
| Layer | Cases that must be tested |
|---|---|
| Fusion A and B (Phase 4, 6) | audio missing, video missing, both missing, unreliable evidence, corrupted evidence; assert output is never "authentic" for both-missing (must be "insufficient evidence") and that a missing branch contributes no score |
| Local pipeline (Phase 5) | same four cases end-to-end on synthetic clips |
| API (Phase 8) | same four cases return structured results/errors, never a fabricated score |
| UI (Phase 9) | each state renders explicitly ("audio missing", "insufficient evidence"); never as a low score |

## Synthetic fixtures
Allowed only for software tests (parsing, status handling, API behavior, UI states). Tiny randomly initialized checkpoints used for API/UI tests carry `is_test_fixture: true` in their model card; the backend echoes it and the UI shows a persistent "test fixture model: not a detector" banner. No fixture output may appear in any results table.

## Frontend honesty rules
The results view renders only data validated against the backend response schema. No mock-score modules in production source (a CI grep check enforces this); progress shown only from real upload events or job status.
