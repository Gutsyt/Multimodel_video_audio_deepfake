"""Score-level fusion (Fusion A): a documented, UNTRAINED rule.

Rule: when TWO modalities are usable (status AVAILABLE), the fused score is the weighted mean of
their raw scores with the weights renormalised over the usable set. Default weights are an
equal-weight prior, NOT fitted on validation data. The flag threshold (default 0.5) is likewise
an unvalidated default operating point.

Missing / corrupted / unreliable / not-analysed modalities are excluded and never contribute a
score (no zero, no one, no default prediction). With exactly ONE usable modality no fusion is
performed: ``fused_score`` is None and the single branch is reported by itself (see
``assessment.py``). With no usable modality the result is INSUFFICIENT_EVIDENCE, never "authentic".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from avdd.common.types import Modality, ModalityEvidence

METHOD = "weighted mean of raw scores over usable modalities (weights renormalised)"
RULE = (
    "Applied only when both the audio and the visual branch produced a usable score: "
    "fused = w_audio * audio_score + w_visual * visual_score, with the weights renormalised to "
    "sum to 1. The default weights (0.5 / 0.5) are an untrained, unvalidated equal-weight prior; the two models "
    "were trained on unrelated data and the rule has not been tested on labelled audio-visual data. "
    "With a single usable branch no fusion is performed."
)


class FusionState(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SINGLE_MODALITY = "single_modality"
    MULTIMODAL = "multimodal"


def _default_weights() -> dict[Modality, float]:
    return {Modality.AUDIO: 0.5, Modality.VISUAL: 0.5}


@dataclass(frozen=True)
class FusionConfig:
    weights: Mapping[Modality, float] = field(default_factory=_default_weights)
    flag_threshold: float = 0.5
    weights_validated: bool = False
    threshold_validated: bool = False

    def __post_init__(self) -> None:
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("fusion weights must be non-negative")
        if not 0.0 <= self.flag_threshold <= 1.0:
            raise ValueError("flag_threshold must be within [0, 1]")


@dataclass(frozen=True)
class FusionResult:
    state: FusionState
    fused_score: float | None
    flagged: bool | None
    weights_used: dict[str, float]
    branch_flags: dict[str, bool]
    excluded: dict[str, str]
    notes: tuple[str, ...]
    threshold: float
    weights_validated: bool
    threshold_validated: bool
    configured_weights: dict[str, float] = field(default_factory=dict)

    @property
    def fusion_performed(self) -> bool:
        return self.state is FusionState.MULTIMODAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "rule": RULE,
            "fusion_performed": self.fusion_performed,
            "configured_weights": {k: round(v, 4) for k, v in self.configured_weights.items()},
            "state": self.state.value,
            "fused_score": None if self.fused_score is None else round(self.fused_score, 4),
            "flagged_at_default_threshold": self.flagged,
            "threshold": self.threshold,
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "branch_flags_at_default_threshold": self.branch_flags,
            "excluded_modalities": self.excluded,
            "weights_validated": self.weights_validated,
            "threshold_validated": self.threshold_validated,
            "calibrated": False,
            "notes": list(self.notes),
        }


def fuse_scores(evidence: Sequence[ModalityEvidence], cfg: FusionConfig | None = None) -> FusionResult:
    """Fuse per-modality evidence. Raises ValueError on duplicate modalities."""
    cfg = cfg or FusionConfig()
    mods = [e.modality for e in evidence]
    if len(set(mods)) != len(mods):
        raise ValueError("evidence contains duplicate modalities")
    usable = [e for e in evidence if e.usable]
    excluded = {e.modality.value: e.status.value for e in evidence if not e.usable}
    notes: list[str] = []
    common: dict[str, Any] = {
        "excluded": excluded, "threshold": cfg.flag_threshold,
        "weights_validated": cfg.weights_validated, "threshold_validated": cfg.threshold_validated,
        "configured_weights": {m.value: float(w) for m, w in cfg.weights.items()},
    }  # fmt: skip
    if not usable:
        notes.append("No modality produced usable evidence. This is NOT evidence that the media is authentic.")
        return FusionResult(FusionState.INSUFFICIENT_EVIDENCE, None, None, {}, {}, notes=tuple(notes), **common)

    flags = {e.modality.value: bool(_score(e) >= cfg.flag_threshold) for e in usable}
    if len(usable) == 1:
        notes.append("Only one modality was usable, so no fusion was performed and no combined score exists. "
                     "The result is that single branch's own score and decision; it is not a multimodal "
                     "result. An excluded modality says nothing about authenticity.")  # fmt: skip
        return FusionResult(FusionState.SINGLE_MODALITY, None, None, {}, flags, notes=tuple(notes), **common)

    raw = {e.modality: float(cfg.weights.get(e.modality, 0.0)) for e in usable}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("weights of usable modalities sum to zero")
    norm = {m: w / total for m, w in raw.items()}
    fused = sum(norm[e.modality] * _score(e) for e in usable)
    if len(set(flags.values())) > 1:
        notes.append("The audio and visual branches disagree at the default threshold; inspect them separately.")
    if not cfg.weights_validated:
        notes.append("Fusion weights are an unvalidated equal-weight default.")
    if not cfg.threshold_validated:
        notes.append("The threshold is an unvalidated default, not an operating point chosen on validation data.")
    return FusionResult(FusionState.MULTIMODAL, fused, fused >= cfg.flag_threshold,
                        {m.value: w for m, w in norm.items()}, flags, notes=tuple(notes), **common)  # fmt: skip


def _score(e: ModalityEvidence) -> float:
    """Score of a USABLE modality. AVAILABLE evidence always carries one (enforced by its type)."""
    if e.score is None:
        raise ValueError(f"{e.modality.value} evidence is usable but has no score")
    return e.score
