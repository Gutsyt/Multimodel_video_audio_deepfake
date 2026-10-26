"""Response schemas (the API contract; also the source of the OpenAPI documentation)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ModalityStatusName = Literal["available", "missing", "corrupted", "unreliable", "not_analyzed"]
JobStatus = Literal["queued", "processing", "completed", "failed"]


class ApiErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ApiErrorBody


class ModelCardOut(BaseModel):
    name: str
    version: str
    modality: str
    license: str
    source_url: str
    citation: str
    trained_on: str
    limitations: list[str]
    is_test_fixture: bool


class MediaInfo(BaseModel):
    kind: Literal["video", "audio", "image"] | None = None
    container: str | None = None
    duration_s: float | None = None
    has_video: bool = False
    has_audio: bool = False
    n_video_streams: int = 0
    n_audio_streams: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class AudioSegment(BaseModel):
    start_s: float
    end_s: float
    score: float
    logits: dict[str, float] | None = None
    class_probs: dict[str, float] | None = None


class FaceBoxOut(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FrameEvidence(BaseModel):
    frame_index: int
    t_s: float | None = None  # None for a still image
    n_faces: int
    face_box: FaceBoxOut | None = None
    face_area_ratio: float | None = None
    crop_px: list[int] | None = None
    score: float | None = None
    logits: dict[str, float] | None = None
    class_probs: dict[str, float] | None = None


class ModalityResult(BaseModel):
    """One modality. ``score`` is a RAW model score and is only present when status is 'available'."""

    status: ModalityStatusName
    score: float | None = None
    reasons: list[str] = []
    aggregation: str | None = None
    runtime_s: float | None = None
    error_code: str | None = None
    decision: Literal["flagged", "not_flagged"] | None = None  # this branch alone, at the default threshold
    source: Literal["video", "image"] | None = None  # visual branch only
    model: ModelCardOut | None = None
    analyzed_limit_s: float | None = None
    diagnostics: dict[str, Any] | None = None  # branch-specific audit data; keys documented in docs/
    # audio evidence
    segments: list[AudioSegment] | None = None
    n_windows: int | None = None
    max_window_score: float | None = None
    metrics: dict[str, float] | None = None
    sample_rate: int | None = None
    # visual evidence
    frames: list[FrameEvidence] | None = None
    n_frames_sampled: int | None = None
    n_face_frames: int | None = None
    face_detector: str | None = None
    median_frame_score: float | None = None
    max_frame_score: float | None = None
    timestamps: str | None = None


class FusionOut(BaseModel):
    method: str
    rule: str | None = None
    fusion_performed: bool | None = None  # None only for results stored before schema 0.2
    configured_weights: dict[str, float] = {}
    state: Literal["insufficient_evidence", "single_modality", "multimodal"]
    fused_score: float | None = None
    flagged_at_default_threshold: bool | None = None
    threshold: float
    weights_used: dict[str, float]
    branch_flags_at_default_threshold: dict[str, bool]
    excluded_modalities: dict[str, str]
    weights_validated: bool
    threshold_validated: bool
    calibrated: bool
    notes: list[str]


class AssessmentOut(BaseModel):
    status: Literal["flagged", "not_flagged", "insufficient_evidence"]
    basis: Literal["multimodal", "audio_only", "visual_only", "none"]
    summary: str
    score: float | None = None
    score_kind: str | None = None
    threshold: float
    threshold_validated: bool
    branch_decisions: dict[str, str | None]
    modalities_used: list[str]
    calibrated: bool
    not_authenticity_note: str


class Interpretation(BaseModel):
    score_meaning: str
    calibrated: bool
    validated_on_project_data: bool
    flag_meaning: str
    not_forensic_proof: str


class AnalysisResult(BaseModel):
    schema_version: str
    avdd_version: str
    created_utc: str
    outcome: Literal["analyzed", "unreadable_media"]
    media: MediaInfo
    modalities: dict[str, ModalityResult]
    fusion: FusionOut
    assessment: AssessmentOut | None = None  # absent in results stored before schema 0.2
    models: list[ModelCardOut]
    is_test_fixture: bool
    interpretation: Interpretation
    analyzed_limit_s: float
    total_runtime_s: float


class AnalysisCreated(BaseModel):
    id: str
    status: JobStatus


class AnalysisStatusOut(BaseModel):
    id: str
    status: JobStatus
    stage: str | None = None
    stages: list[str]
    created_utc: str
    updated_utc: str
    result: AnalysisResult | None = None
    error: ApiErrorBody | None = None


class AnalysisSummary(BaseModel):
    id: str
    status: JobStatus
    created_utc: str
    duration_s: float | None = None
    media_kind: str | None = None
    assessment_status: str | None = None
    basis: str | None = None
    audio_decision: str | None = None
    visual_decision: str | None = None
    fusion_state: str | None = None
    fused_score: float | None = None
    flagged: bool | None = None
    audio_status: str | None = None
    audio_score: float | None = None
    visual_status: str | None = None
    visual_score: float | None = None
    is_test_fixture: bool | None = None
    error_code: str | None = None


class HistoryOut(BaseModel):
    items: list[AnalysisSummary]


class ClearedOut(BaseModel):
    deleted: int


class ModelStatusOut(BaseModel):
    role: Literal["audio", "visual"]
    state: Literal["loading", "loaded", "failed", "disabled"]
    error_code: str | None = None
    card: ModelCardOut | None = None


class AcceptedKind(BaseModel):
    label: str
    extensions: list[str]


class LimitsOut(BaseModel):
    max_upload_mb: float
    max_duration_s: float
    max_image_megapixels: float | None = None
    accepted_containers: list[str]
    accepted: dict[str, AcceptedKind] = {}
    frames_sampled: int
    max_pending_jobs: int


class CapabilitiesOut(BaseModel):
    avdd_version: str
    models_state: Literal["loading", "ready", "unavailable"]
    models: list[ModelStatusOut]
    limits: LimitsOut
    retention: str
    evaluation_status: str
    stages: list[str]
