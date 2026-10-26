"""Visual branch: sample frames -> detect largest face -> crop -> detector -> aggregate."""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from avdd.common.branch import BranchOutput
from avdd.common.errors import VideoError
from avdd.common.protocols import FaceImageDetector
from avdd.common.types import Modality, ModalityEvidence, ModalityStatus
from avdd.video.faces import FaceBoxDetector, crop_face_rgb
from avdd.video.frames import sample_frames

logger = logging.getLogger(__name__)
AGGREGATION = "mean of per-frame raw scores over sampled frames with a detected face"


@dataclass(frozen=True)
class VideoBranchConfig:
    n_frames: int = 16
    max_seconds: float = 60.0
    min_face_frames: int = 4
    min_face_fraction: float = 0.3
    face_margin: float = 0.3


def run_video_branch(
    path: str | Path,
    detector: FaceImageDetector,
    face_detector: FaceBoxDetector,
    cfg: VideoBranchConfig | None = None,
) -> BranchOutput:
    """Analyse sampled frames. No face / too few faces => UNRELIABLE, never a 'real' score."""
    cfg = cfg or VideoBranchConfig()
    t0 = perf_counter()
    card = detector.model_card.to_dict()
    try:
        frames = sample_frames(path, cfg.n_frames, cfg.max_seconds)
    except VideoError as exc:
        ev = ModalityEvidence(Modality.VISUAL, ModalityStatus.CORRUPTED,
                              reasons=(f"{exc.code}: {exc.message}",))  # fmt: skip
        return BranchOutput(ev, {"error_code": exc.code, "model": card})

    infos: list[dict[str, Any]] = []
    crops, crop_pos = [], []
    for fr in frames:
        faces = face_detector.detect(fr.image_bgr)
        info: dict[str, Any] = {"frame_index": fr.index, "t_s": round(fr.t_s, 3), "n_faces": len(faces)}
        if faces:
            b = faces[0]
            info["face_box"] = {"x": b.x, "y": b.y, "w": b.w, "h": b.h}
            crops.append(crop_face_rgb(fr.image_bgr, b, cfg.face_margin))
            crop_pos.append(len(infos))
        infos.append(info)

    details: dict[str, Any] = {
        "model": card, "aggregation": AGGREGATION, "face_detector": face_detector.name,
        "n_frames_sampled": len(frames), "n_face_frames": len(crops), "frames": infos,
        "timestamps": "nominal (frame index / fps)",
    }  # fmt: skip
    needed = max(cfg.min_face_frames, math.ceil(cfg.min_face_fraction * len(frames)))
    if len(crops) < needed:
        reason = f"faces found in {len(crops)} of {len(frames)} sampled frames (need at least {needed})"
        details["runtime_s"] = round(perf_counter() - t0, 3)
        return BranchOutput(ModalityEvidence(Modality.VISUAL, ModalityStatus.UNRELIABLE,
                                             reasons=(reason,)), details)  # fmt: skip

    try:
        scores = detector.score_faces(crops)
        if len(scores) != len(crops) or not all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores):
            raise ValueError("detector returned invalid scores")
    except Exception:
        logger.exception("visual model inference failed")
        return BranchOutput(ModalityEvidence(Modality.VISUAL, ModalityStatus.NOT_ANALYZED,
                                             reasons=("visual model inference failed",)), details)  # fmt: skip

    for pos, s in zip(crop_pos, scores, strict=True):
        infos[pos]["score"] = round(s, 4)
    caveats: list[str] = []
    multi = sum(1 for i in infos if i["n_faces"] > 1)
    if multi:
        caveats.append(f"multiple faces in {multi} sampled frame(s); only the largest face was analysed")
    details["median_frame_score"] = round(statistics.median(scores), 4)
    details["max_frame_score"] = round(max(scores), 4)
    details["runtime_s"] = round(perf_counter() - t0, 3)
    mean = sum(scores) / len(scores)
    return BranchOutput(ModalityEvidence(Modality.VISUAL, ModalityStatus.AVAILABLE,
                                         score=mean, reasons=tuple(caveats)), details)  # fmt: skip
