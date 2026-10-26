"""Audio branch: decode -> usability gate -> windows -> detector -> aggregate."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from time import perf_counter

from avdd.audio.extract import SAMPLE_RATE, assess_audio, decode_audio
from avdd.audio.windows import make_windows
from avdd.common.branch import BranchOutput
from avdd.common.errors import AudioError
from avdd.common.protocols import AudioWindowDetector
from avdd.common.types import Modality, ModalityEvidence, ModalityStatus

logger = logging.getLogger(__name__)
AGGREGATION = "mean of per-window raw scores"


def run_audio_branch(
    path: str | Path, detector: AudioWindowDetector, max_seconds: float = 60.0
) -> BranchOutput:
    """Analyse the audio track. Never converts a failure into a 'real' score."""
    t0 = perf_counter()
    card = detector.model_card.to_dict()
    try:
        wave = decode_audio(path, max_seconds=max_seconds)
    except AudioError as exc:
        ev = ModalityEvidence(Modality.AUDIO, ModalityStatus.CORRUPTED,
                              reasons=(f"{exc.code}: {exc.message}",))  # fmt: skip
        return BranchOutput(ev, {"error_code": exc.code, "model": card})

    reasons, metrics = assess_audio(wave)
    details: dict[str, object] = {"metrics": metrics, "model": card, "aggregation": AGGREGATION,
                                  "analyzed_limit_s": max_seconds}  # fmt: skip
    if reasons:
        ev = ModalityEvidence(Modality.AUDIO, ModalityStatus.UNRELIABLE, reasons=tuple(reasons))
        details["runtime_s"] = round(perf_counter() - t0, 3)
        return BranchOutput(ev, details)

    windows = make_windows(wave)
    try:
        scores = detector.score_windows([w.samples for w in windows])
        if len(scores) != len(windows) or not all(
            math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores
        ):
            raise ValueError("detector returned invalid scores")
    except Exception:
        logger.exception("audio model inference failed")
        ev = ModalityEvidence(Modality.AUDIO, ModalityStatus.NOT_ANALYZED,
                              reasons=("audio model inference failed",))  # fmt: skip
        return BranchOutput(ev, details)

    details["segments"] = [
        {"start_s": round(w.start_s, 3), "end_s": round(w.end_s, 3), "score": round(s, 4)}
        for w, s in zip(windows, scores, strict=True)
    ]
    details["n_windows"] = len(windows)
    details["max_window_score"] = round(max(scores), 4)
    details["sample_rate"] = SAMPLE_RATE
    details["runtime_s"] = round(perf_counter() - t0, 3)
    mean = sum(scores) / len(scores)
    caveats = ("only one analysis window", ) if len(windows) == 1 else ()
    return BranchOutput(ModalityEvidence(Modality.AUDIO, ModalityStatus.AVAILABLE,
                                         score=mean, reasons=caveats), details)  # fmt: skip
