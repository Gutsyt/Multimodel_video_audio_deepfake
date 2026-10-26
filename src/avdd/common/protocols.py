"""Structural interfaces so detectors can be swapped (or stubbed in software tests)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from avdd.common.model_card import ModelCard


@dataclass(frozen=True)
class ScoredItem:
    """One model output with the evidence needed to audit it.

    ``logits`` and ``probs`` map class label -> value for EVERY class the network outputs, so a
    reader can see both sides of the softmax and the logit margin, not only the reported class.
    They are ``None`` for detectors (e.g. test doubles) that only return a score.
    """

    score: float
    logits: dict[str, float] | None = None
    probs: dict[str, float] | None = None


class AudioWindowDetector(Protocol):
    """Scores fixed-length 16 kHz mono windows; returns raw P(spoof)-like scores in [0, 1]."""

    model_card: ModelCard

    def score_windows(self, windows: Sequence[NDArray[np.float32]]) -> list[float]: ...


class FaceImageDetector(Protocol):
    """Scores RGB uint8 face crops; returns raw P(fake)-like scores in [0, 1]."""

    model_card: ModelCard

    def score_faces(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[float]: ...


def output_semantics_of(detector: object) -> dict[str, Any] | None:
    """Optional ``output_semantics`` attribute: which class the reported score is, and why."""
    value = getattr(detector, "output_semantics", None)
    return dict(value) if isinstance(value, dict) else None


def device_of(detector: object) -> str | None:
    value = getattr(detector, "device", None)
    return value if isinstance(value, str) else None
