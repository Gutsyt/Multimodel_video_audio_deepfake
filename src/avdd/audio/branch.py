"""Audio branch: decode -> usability gate -> windows -> detector -> aggregate."""

from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from avdd.audio.extract import SAMPLE_RATE, assess_audio, audio_input_diagnostics, decode_audio
from avdd.audio.windows import WINDOW_SAMPLES, make_windows
from avdd.common.branch import BranchOutput
from avdd.common.errors import AudioError
from avdd.common.protocols import (
    AudioWindowDetector,
    ScoredItem,
    device_of,
    output_semantics_of,
)
from avdd.common.types import Modality, ModalityEvidence, ModalityStatus

logger = logging.getLogger(__name__)
AGGREGATION = "mean of per-window raw scores"
SATURATION_LEVEL = 0.99  # a window at or above this is "saturated": the softmax says nothing finer
DISAGREEMENT_RANGE = 0.5  # max-min across windows at/above this is called out; it changes no score


def _score_items(detector: AudioWindowDetector, windows: list[Any]) -> list[ScoredItem]:
    """Use the detailed scorer when the detector offers one; otherwise plain scores."""
    detailed = getattr(detector, "score_windows_detailed", None)
    if callable(detailed):
        items: list[ScoredItem] = detailed(windows)
        return items
    return [ScoredItem(score=s) for s in detector.score_windows(windows)]


def _window_stats(scores: list[float], threshold: float) -> dict[str, Any]:
    return {
        "mean": round(sum(scores) / len(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "n_at_or_above_threshold": sum(1 for s in scores if s >= threshold),
        "n_saturated": sum(1 for s in scores if s >= SATURATION_LEVEL),
        "range": round(max(scores) - min(scores), 4),
        "threshold": threshold,
    }


def _tail_note(n_samples: int, n_windows: int) -> str:
    full, rem = divmod(n_samples, WINDOW_SAMPLES)
    if rem == 0:
        return "audio length is an exact multiple of the window; nothing tiled or dropped"
    if n_windows == full:
        return f"a trailing piece of {rem / SAMPLE_RATE:.1f} s (under 1 s) was dropped"
    return f"the last window holds {rem / SAMPLE_RATE:.1f} s of real audio, tiled to full length"


def run_audio_branch(
    path: str | Path,
    detector: AudioWindowDetector,
    max_seconds: float = 60.0,
    threshold: float = 0.5,
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
    input_diag = audio_input_diagnostics(wave, max_seconds)
    input_diag.update({"window_samples": WINDOW_SAMPLES, "window_s": round(WINDOW_SAMPLES / SAMPLE_RATE, 3),
                       "hop": "equal to the window (no overlap)", "device": device_of(detector)})  # fmt: skip
    diagnostics: dict[str, Any] = {"input": input_diag, "output_semantics": output_semantics_of(detector)}
    details: dict[str, Any] = {"metrics": metrics, "model": card, "aggregation": AGGREGATION,
                               "analyzed_limit_s": max_seconds, "diagnostics": diagnostics}  # fmt: skip
    if reasons:
        ev = ModalityEvidence(Modality.AUDIO, ModalityStatus.UNRELIABLE, reasons=tuple(reasons))
        details["runtime_s"] = round(perf_counter() - t0, 3)
        return BranchOutput(ev, details)

    windows = make_windows(wave)
    try:
        items = _score_items(detector, [w.samples for w in windows])
        scores = [it.score for it in items]
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
        {"start_s": round(w.start_s, 3), "end_s": round(w.end_s, 3), "score": round(it.score, 4),
         "logits": _rounded(it.logits), "class_probs": _rounded(it.probs, 5)}
        for w, it in zip(windows, items, strict=True)
    ]
    diagnostics["windows"] = {**_window_stats(scores, threshold),
                              "n_windows": len(windows), "tail": _tail_note(int(wave.shape[0]), len(windows))}  # fmt: skip
    details["n_windows"] = len(windows)
    details["max_window_score"] = round(max(scores), 4)
    details["sample_rate"] = SAMPLE_RATE
    details["runtime_s"] = round(perf_counter() - t0, 3)
    mean = sum(scores) / len(scores)
    caveats: list[str] = ["only one analysis window"] if len(windows) == 1 else []
    if len(scores) > 1 and max(scores) - min(scores) >= DISAGREEMENT_RANGE:
        caveats.append(
            f"window scores disagree strongly ({min(scores):.2f} to {max(scores):.2f}); "
            "the mean hides this, so read the per-window values"
        )
    return BranchOutput(ModalityEvidence(Modality.AUDIO, ModalityStatus.AVAILABLE,
                                         score=mean, reasons=tuple(caveats)), details)  # fmt: skip


def _rounded(values: dict[str, float] | None, digits: int = 3) -> dict[str, float] | None:
    return None if values is None else {k: round(v, digits) for k, v in values.items()}
