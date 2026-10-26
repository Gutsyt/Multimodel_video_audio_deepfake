"""Model lifecycle: load pretrained detectors in the background; report their real state."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from avdd.api.settings import Settings
from avdd.common.errors import AvddError
from avdd.common.protocols import AudioWindowDetector, FaceImageDetector
from avdd.video.faces import FaceBoxDetector

logger = logging.getLogger(__name__)
ModelsState = Literal["loading", "ready", "unavailable"]


@dataclass
class ModelStatus:
    role: Literal["audio", "visual"]
    state: Literal["loading", "loaded", "failed", "disabled"]
    error_code: str | None = None
    card: dict[str, Any] | None = None


@dataclass
class DetectorBundle:
    audio: AudioWindowDetector | None = None
    video: FaceImageDetector | None = None
    face: FaceBoxDetector | None = None
    statuses: list[ModelStatus] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.audio is not None or (self.video is not None and self.face is not None)


def bundle_from_detectors(
    audio: AudioWindowDetector | None,
    video: FaceImageDetector | None,
    face: FaceBoxDetector | None,
    failures: dict[str, str] | None = None,
) -> DetectorBundle:
    """Build a bundle and derive per-role statuses (``failures`` maps role -> error code)."""
    failures = failures or {}
    st: list[ModelStatus] = []
    for role, det in (("audio", audio), ("visual", video)):
        r: Literal["audio", "visual"] = "audio" if role == "audio" else "visual"
        if r in failures:
            st.append(ModelStatus(r, "failed", failures[r]))
        elif det is None:
            st.append(ModelStatus(r, "disabled"))
        else:
            st.append(ModelStatus(r, "loaded", None, det.model_card.to_dict()))
    return DetectorBundle(audio, video, face, st)


class DetectorManager:
    """Holds the loaded detectors. Real models load in a background thread."""

    def __init__(self, settings: Settings, injected: DetectorBundle | None = None) -> None:
        self._settings = settings
        self._bundle: DetectorBundle | None = injected
        self._state: ModelsState = "loading"
        self._lock = threading.Lock()
        if injected is not None:
            self._state = "ready" if injected.usable else "unavailable"

    def start(self) -> None:
        if self._bundle is None:
            threading.Thread(target=self._load_real, name="avdd-model-loader", daemon=True).start()

    @property
    def state(self) -> ModelsState:
        with self._lock:
            return self._state

    def bundle(self) -> DetectorBundle | None:
        with self._lock:
            return self._bundle

    def statuses(self) -> list[ModelStatus]:
        b = self.bundle()
        if b is not None:
            return b.statuses
        return [ModelStatus("audio", "loading"), ModelStatus("visual", "loading")]

    def _load_real(self) -> None:
        s = self._settings
        audio: AudioWindowDetector | None = None
        video: FaceImageDetector | None = None
        face: FaceBoxDetector | None = None
        failures: dict[str, str] = {}
        try:
            if s.audio_model != "none":
                from avdd.audio.aasist_detector import AasistDetector

                a = AasistDetector(s.audio_model, device=s.device)
                try:
                    a.load()
                    audio = a
                except AvddError as exc:
                    failures["audio"] = exc.code
                    logger.warning("audio model not loaded: %s", exc.code)
            if s.video_model == "hf":
                from avdd.video.faces import HaarFaceDetector
                from avdd.video.hf_detector import (
                    DEFAULT_MODEL_ID,
                    DEFAULT_REVISION,
                    HfImageDetector,
                )

                v = HfImageDetector(s.hf_model_id or DEFAULT_MODEL_ID, s.hf_revision or DEFAULT_REVISION, s.device)
                try:
                    face = HaarFaceDetector()
                    info = v.load()
                    video = v
                    logger.info("visual model loaded: %s", info)
                except AvddError as exc:
                    failures["visual"] = exc.code
                    logger.warning("visual model not loaded: %s", exc.code)
                except Exception:
                    failures["visual"] = "face_detector_unavailable"
                    logger.warning("face detector unavailable", exc_info=True)
        except Exception:
            logger.exception("unexpected error while loading models")
            failures.setdefault("audio", "load_error")
        bundle = bundle_from_detectors(audio, video, face, failures)
        with self._lock:
            self._bundle = bundle
            self._state = "ready" if bundle.usable else "unavailable"
