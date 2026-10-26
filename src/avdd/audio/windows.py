"""Split a waveform into the fixed-length windows AASIST expects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from avdd.audio.extract import SAMPLE_RATE

WINDOW_SAMPLES = 64_600  # ~4.04 s at 16 kHz; fixed by the pretrained AASIST configuration


@dataclass(frozen=True)
class AudioWindow:
    start_s: float
    end_s: float
    samples: NDArray[np.float32]


def make_windows(
    wave: NDArray[np.float32], window: int = WINDOW_SAMPLES, min_tail: int = SAMPLE_RATE
) -> list[AudioWindow]:
    """Non-overlapping windows. A short final piece is tiled to full length (as upstream
    AASIST evaluation does); a trailing piece shorter than ``min_tail`` is dropped unless
    it is the only piece."""
    n = int(wave.shape[0])
    out: list[AudioWindow] = []
    pos = 0
    while pos < n:
        seg = wave[pos : pos + window]
        if seg.shape[0] < window:
            if out and seg.shape[0] < min_tail:
                break
            seg = np.tile(seg, math.ceil(window / seg.shape[0]))[:window]
        out.append(AudioWindow(pos / SAMPLE_RATE, min(pos + window, n) / SAMPLE_RATE,
                               np.ascontiguousarray(seg, dtype=np.float32)))  # fmt: skip
        pos += window
    return out
