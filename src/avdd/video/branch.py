"""Visual branch: sample frames (or load one image) -> detect largest face -> crop -> detector -> aggregate."""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from avdd.common.branch import BranchOutput
from avdd.common.errors import VideoError
from avdd.common.protocols import FaceImageDetector, ScoredItem, device_of, output_semantics_of
from avdd.common.types import Modality, ModalityEvidence, ModalityStatus
from avdd.video.faces import FaceBoxDetector, crop_face_rgb
from avdd.video.frames import SampledFrame, load_image_frame, sample_frames

logger = logging.getLogger(__name__)
AGGREGATION = "median of per-frame calibrated scores over sampled frames with a detected face"
IMAGE_AGGREGATION = "calibrated score of the largest detected face in the image"
SATURATION_LEVEL = 0.99
DISAGREEMENT_RANGE = 0.5  # max-min across frames at/above this is called out; it changes no score


@dataclass(frozen=True)
class VideoBranchConfig:
    n_frames: int = 12
    max_seconds: float = 60.0
    min_face_frames: int = 4
    min_face_fraction: float = 0.3
    face_margin: float = 0.3
    threshold: float = 0.5  # only used to count frames at/above it in the diagnostics


def run_video_branch(
    path: str | Path,
    detector: FaceImageDetector,
    face_detector: FaceBoxDetector,
    cfg: VideoBranchConfig | None = None,
) -> BranchOutput:
    """Analyse sampled frames. No face / too few faces => UNRELIABLE, never a 'real' score."""
    cfg = cfg or VideoBranchConfig()
    card = detector.model_card.to_dict()
    try:
        frames = sample_frames(path, cfg.n_frames, cfg.max_seconds)
    except VideoError as exc:
        return _corrupted(exc, card)
    return _score_frames(frames, detector, face_detector, cfg, source="video", aggregation=AGGREGATION)


def run_image_branch(
    path: str | Path,
    detector: FaceImageDetector,
    face_detector: FaceBoxDetector,
    cfg: VideoBranchConfig | None = None,
) -> BranchOutput:
    """Analyse one still image: needs a single detected face; there is no temporal evidence."""
    cfg = replace(cfg or VideoBranchConfig(), min_face_frames=1, min_face_fraction=0.0)
    card = detector.model_card.to_dict()
    try:
        frame = load_image_frame(path)
    except VideoError as exc:
        return _corrupted(exc, card)
    return _score_frames([frame], detector, face_detector, cfg, source="image",
                         aggregation=IMAGE_AGGREGATION)  # fmt: skip


def _corrupted(exc: VideoError, card: dict[str, Any]) -> BranchOutput:
    ev = ModalityEvidence(Modality.VISUAL, ModalityStatus.CORRUPTED,
                          reasons=(f"{exc.code}: {exc.message}",))  # fmt: skip
    return BranchOutput(ev, {"error_code": exc.code, "model": card})


def _score_items(detector: FaceImageDetector, crops: Sequence[Any]) -> list[ScoredItem]:
    detailed = getattr(detector, "score_faces_detailed", None)
    if callable(detailed):
        items: list[ScoredItem] = detailed(crops)
        return items
    return [ScoredItem(score=s) for s in detector.score_faces(crops)]


def _score_frames(
    frames: Sequence[SampledFrame],
    detector: FaceImageDetector,
    face_detector: FaceBoxDetector,
    cfg: VideoBranchConfig,
    source: str,
    aggregation: str,
) -> BranchOutput:
    t0 = perf_counter()
    card = detector.model_card.to_dict()
    fh, fw = frames[0].image_bgr.shape[:2]
    infos: list[dict[str, Any]] = []
    crops, crop_pos = [], []
    for fr in frames:
        faces = face_detector.detect(fr.image_bgr)
        h, w = fr.image_bgr.shape[:2]
        info: dict[str, Any] = {"frame_index": fr.index,
                                "t_s": None if fr.t_s is None else round(fr.t_s, 3),
                                "n_faces": len(faces)}  # fmt: skip
        if faces:
            b = faces[0]
            info["face_box"] = {"x": b.x, "y": b.y, "w": b.w, "h": b.h}
            info["face_area_ratio"] = round(b.area / float(w * h), 4)
            crop = crop_face_rgb(fr.image_bgr, b, cfg.face_margin)
            info["crop_px"] = [int(crop.shape[1]), int(crop.shape[0])]
            crops.append(crop)
            crop_pos.append(len(infos))
        infos.append(info)

    needed = max(cfg.min_face_frames, math.ceil(cfg.min_face_fraction * len(frames)))
    diagnostics: dict[str, Any] = {
        "source": source,
        "input": {
            "frame_width": fw, "frame_height": fh, "face_detector": face_detector.name,
            "face_selection": "largest detected face in each frame; other faces are ignored",
            "crop_rule": f"square box around the face expanded by {cfg.face_margin:g} per side",
            "min_face_frames_required": needed, "device": device_of(detector),
        },
        "output_semantics": output_semantics_of(detector),
    }  # fmt: skip
    details: dict[str, Any] = {
        "model": card, "aggregation": aggregation, "face_detector": face_detector.name,
        "source": source, "n_frames_sampled": len(frames), "n_face_frames": len(crops), "frames": infos,
        "timestamps": "none (still image)" if source == "image" else "nominal (frame index / fps)",
        "diagnostics": diagnostics,
    }  # fmt: skip
    if len(crops) < needed:
        what = "the image" if source == "image" else f"{len(crops)} of {len(frames)} sampled frames"
        reason = (f"no face was detected in {what}" if source == "image"
                  else f"faces found in {len(crops)} of {len(frames)} sampled frames (need at least {needed})")  # fmt: skip
        details["runtime_s"] = round(perf_counter() - t0, 3)
        return BranchOutput(ModalityEvidence(Modality.VISUAL, ModalityStatus.UNRELIABLE,
                                             reasons=(reason,)), details)  # fmt: skip

    try:
        items = _score_items(detector, crops)
        scores = [it.score for it in items]
        if len(scores) != len(crops) or not all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores):
            raise ValueError("detector returned invalid scores")
    except Exception:
        logger.exception("visual model inference failed")
        return BranchOutput(ModalityEvidence(Modality.VISUAL, ModalityStatus.NOT_ANALYZED,
                                             reasons=("visual model inference failed",)), details)  # fmt: skip

    for pos, it in zip(crop_pos, items, strict=True):
        infos[pos]["score"] = round(it.score, 4)
    caveats: list[str] = []
    multi = sum(1 for i in infos if i["n_faces"] > 1)
    if multi:
        where = "the image" if source == "image" else f"{multi} sampled frame(s)"
        caveats.append(f"multiple faces in {where}; only the largest face was analysed")
    diagnostics["scores"] = {
        "mean": round(sum(scores) / len(scores), 4), "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4), "max": round(max(scores), 4),
        "std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "n_at_or_above_threshold": sum(1 for s in scores if s >= cfg.threshold),
        "n_saturated": sum(1 for s in scores if s >= SATURATION_LEVEL),
        "range": round(max(scores) - min(scores), 4),
        "threshold": cfg.threshold,
    }
    if len(scores) > 1 and max(scores) - min(scores) >= DISAGREEMENT_RANGE:
        caveats.append(
            f"frame scores disagree strongly ({min(scores):.2f} to {max(scores):.2f}); "
            "the median limits their effect, but read the per-frame values"
        )
    details["median_frame_score"] = round(statistics.median(scores), 4)
    details["max_frame_score"] = round(max(scores), 4)
    details["runtime_s"] = round(perf_counter() - t0, 3)
    return BranchOutput(ModalityEvidence(Modality.VISUAL, ModalityStatus.AVAILABLE,
                                         score=statistics.median(scores), reasons=tuple(caveats)), details)  # fmt: skip
