"""Detector stubs for SOFTWARE tests. They return fixed numbers and say nothing about detection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from avdd.common.model_card import ModelCard
from avdd.common.types import Modality
from avdd.video.faces import FaceBox


def _card(m: Modality) -> ModelCard:
    return ModelCard("stub", "test", m, "n/a", "n/a", "n/a", "nothing", ("test fixture",), is_test_fixture=True)


class StubAudio:
    def __init__(self, score: float = 0.7, mode: str = "ok") -> None:
        self.model_card = _card(Modality.AUDIO)
        self.score, self.mode, self.calls = score, mode, 0

    def score_windows(self, windows: Sequence[NDArray[np.float32]]) -> list[float]:
        self.calls += 1
        if self.mode == "raise":
            raise RuntimeError("boom with /secret/path")
        if self.mode == "nan":
            return [float("nan")] * len(windows)
        return [self.score] * len(windows)


class StubFace:
    def __init__(self, score: float = 0.3, mode: str = "ok") -> None:
        self.model_card = _card(Modality.VISUAL)
        self.score, self.mode, self.calls = score, mode, 0

    def score_faces(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[float]:
        self.calls += 1
        if self.mode == "raise":
            raise RuntimeError("boom")
        return [self.score] * len(faces_rgb)


class StubBoxes:
    name = "stub-boxes"

    def __init__(self, n_faces: int = 1) -> None:
        self.n_faces = n_faces

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[FaceBox]:
        h, w = image_bgr.shape[:2]
        return [FaceBox(w // 4 + i, h // 4 + i, w // 2, h // 2) for i in range(self.n_faces)]


class BlockingAudio(StubAudio):
    """Test double that pauses inside inference so tests can observe real in-flight job state."""

    def __init__(self, score: float = 0.7) -> None:
        super().__init__(score)
        import threading

        self.entered = threading.Event()
        self.release = threading.Event()

    def score_windows(self, windows: Sequence[NDArray[np.float32]]) -> list[float]:
        self.entered.set()
        self.release.wait(timeout=30)
        return super().score_windows(windows)
