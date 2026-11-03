"""End-to-end analysis of one media file -> JSON-serializable result (no file paths inside)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from avdd import __version__
from avdd.audio.branch import run_audio_branch
from avdd.common.branch import BranchOutput
from avdd.common.errors import AvddError
from avdd.common.protocols import AudioWindowDetector, FaceImageDetector
from avdd.common.types import Modality, ModalityEvidence, ModalityStatus
from avdd.fusion.assessment import build_assessment, thresholds_of
from avdd.media.availability import availability_from_probe
from avdd.media.probe import ProbeStatus, probe_media
from avdd.video.branch import VideoBranchConfig, run_image_branch, run_video_branch
from avdd.video.faces import FaceBoxDetector

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"
STAGES: tuple[str, ...] = ("probing", "audio", "visual", "verdict")  # reported via on_stage


@dataclass(frozen=True)
class AnalysisConfig:
    max_seconds: float = 60.0
    video: VideoBranchConfig = field(default_factory=VideoBranchConfig)


def _absent(modality: Modality, status: ModalityStatus, reason: str) -> BranchOutput:
    reasons = (reason,) if reason else ()
    return BranchOutput(ModalityEvidence(modality, status, reasons=reasons), {})


def _missing_reason(modality: Modality, kind: str | None) -> str:
    """Why a stream is absent, in terms of what the user uploaded."""
    if modality is Modality.AUDIO:
        return "this image has no audio" if kind == "image" else "no audio track in this file"
    return "this audio file has no video stream" if kind == "audio" else "no video stream in this file"


def _modality_json(out: BranchOutput) -> dict[str, Any]:
    e = out.evidence
    return {"status": e.status.value, "score": None if e.score is None else round(e.score, 4),
            "reasons": list(e.reasons), **out.details}  # fmt: skip


def analyze_media(
    path: str | Path,
    audio_detector: AudioWindowDetector | None,
    video_detector: FaceImageDetector | None,
    face_detector: FaceBoxDetector | None,
    cfg: AnalysisConfig | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Analyse ``path``. Raises ``AvddError`` only for caller/environment problems
    (input_not_found, tool_missing); unreadable media yields a result with outcome
    'unreadable_media' and CORRUPTED modalities."""
    cfg = cfg or AnalysisConfig()
    notify: Callable[[str], None] = on_stage or (lambda _stage: None)
    t0 = perf_counter()
    notify("probing")
    probe = probe_media(path)
    if probe.status in (ProbeStatus.NOT_FOUND, ProbeStatus.TOOL_MISSING):
        raise AvddError("input_not_found" if probe.status is ProbeStatus.NOT_FOUND else "tool_missing",
                        probe.error or probe.status.value)  # fmt: skip

    avail = availability_from_probe(probe)
    outcome = "analyzed" if probe.status is ProbeStatus.OK else "unreadable_media"
    results: dict[Modality, BranchOutput] = {}
    th = {Modality.AUDIO: thresholds_of(audio_detector), Modality.VISUAL: thresholds_of(video_detector)}

    for modality in (Modality.AUDIO, Modality.VISUAL):
        status = avail[modality]
        if status is ModalityStatus.CORRUPTED:
            results[modality] = _absent(modality, status, "container could not be parsed")
        elif status is ModalityStatus.MISSING:
            results[modality] = _absent(modality, status, _missing_reason(modality, probe.kind))
        elif modality is Modality.AUDIO:
            if audio_detector is None:
                results[modality] = _absent(modality, ModalityStatus.NOT_ANALYZED, "no audio model loaded")
            else:
                notify("audio")
                results[modality] = run_audio_branch(path, audio_detector, cfg.max_seconds, th[modality].t_fake)
        elif video_detector is None or face_detector is None:
            results[modality] = _absent(modality, ModalityStatus.NOT_ANALYZED, "no visual model loaded")
        else:
            notify("visual")
            vcfg = replace(cfg.video, threshold=th[modality].t_fake)
            run = run_image_branch if probe.kind == "image" else run_video_branch
            results[modality] = run(path, video_detector, face_detector, vcfg)

    notify("verdict")
    evidence = [results[Modality.AUDIO].evidence, results[Modality.VISUAL].evidence]
    assessment = build_assessment(evidence, th)
    cards = [d.model_card.to_dict() for d in (audio_detector, video_detector) if d is not None]
    media = {k: v for k, v in probe.to_dict().items() if k not in ("status", "error")}
    return {
        "schema_version": SCHEMA_VERSION,
        "avdd_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome": outcome,
        "media": media,
        "modalities": {m.value: {**_modality_json(o), "verdict": assessment["branch_verdicts"][m.value]}
                       for m, o in results.items()},
        "assessment": assessment,
        "models": cards,
        "is_test_fixture": any(c["is_test_fixture"] for c in cards),
        "analyzed_limit_s": cfg.max_seconds,
        "total_runtime_s": round(perf_counter() - t0, 3),
    }
