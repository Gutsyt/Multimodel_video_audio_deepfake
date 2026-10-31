"""Split a waveform into fixed 4 s windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from avdd.audio.extract import SAMPLE_RATE

WINDOW_SAMPLES = 64_000  # 4 s at 16 kHz: the window length the audio head was trained on


@dataclass(frozen=True)
class AudioWindow:
    start_s: float
    end_s: float
    samples: NDArray[np.float32]
    synthetic: bool = False  # True when the window contains looped (tiled) audio that never existed


def make_windows(
    wave: NDArray[np.float32], window: int = WINDOW_SAMPLES, min_tail: int = SAMPLE_RATE
) -> list[AudioWindow]:
    """Windows made only of audio that is really in the file, except for very short clips.

    * Clip at least one window long: non-overlapping full windows, plus (when the leftover is at
      least ``min_tail`` samples) one final window that is the LAST ``window`` samples of the clip,
      so it overlaps the previous window instead of containing invented audio. A leftover shorter
      than ``min_tail`` is dropped.
    * Clip shorter than one window: tiled to full length, which is what the upstream AASIST
      evaluation does for short utterances (marked ``synthetic``).

    Why not tile a trailing piece of a longer clip: on 73 genuine LibriSpeech clips, tiled trailing
    windows were scored as spoof 49% of the time (20 of 41), against 6% (4 of 63) for natural windows.
    Looping a mid-speech fragment builds a repeating signal with hard seams that the detector reads
    as an artefact, and that window then counted as much as any real one in the mean.
    """
    n = int(wave.shape[0])
    if n == 0:
        return []
    if n < window:
        seg = np.tile(wave, -(-window // n))[:window]
        return [AudioWindow(0.0, n / SAMPLE_RATE, np.ascontiguousarray(seg, dtype=np.float32), True)]
    out = [
        AudioWindow(pos / SAMPLE_RATE, (pos + window) / SAMPLE_RATE,
                    np.ascontiguousarray(wave[pos : pos + window], dtype=np.float32))
        for pos in range(0, n - window + 1, window)
    ]  # fmt: skip
    leftover = n - (n // window) * window
    if leftover >= min_tail:
        out.append(AudioWindow((n - window) / SAMPLE_RATE, n / SAMPLE_RATE,
                               np.ascontiguousarray(wave[-window:], dtype=np.float32)))  # fmt: skip
    return out
