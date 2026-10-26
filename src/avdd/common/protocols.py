"""Structural interfaces so detectors can be swapped (or stubbed in software tests)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from avdd.common.model_card import ModelCard


class AudioWindowDetector(Protocol):
    """Scores fixed-length 16 kHz mono windows; returns raw P(spoof)-like scores in [0, 1]."""

    model_card: ModelCard

    def score_windows(self, windows: Sequence[NDArray[np.float32]]) -> list[float]: ...


class FaceImageDetector(Protocol):
    """Scores RGB uint8 face crops; returns raw P(fake)-like scores in [0, 1]."""

    model_card: ModelCard

    def score_faces(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[float]: ...
