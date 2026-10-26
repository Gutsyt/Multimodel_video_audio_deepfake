"""Score-level fusion (Fusion A): a documented, UNTRAINED rule.

Rule: weighted mean of the raw scores of USABLE modalities only (status AVAILABLE), with
weights renormalised over the usable set. Default weights are an equal-weight prior, NOT
fitted on validation data. The flag threshold (default 0.5) is likewise an unvalidated
default operating point. Missing / corrupted / unreliable / not-analysed modalities are
excluded and never contribute a score; with no usable modality the result is
INSUFFICIENT_EVIDENCE, never "authentic".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from avdd.common.types import Modality, ModalityEvidence

METHOD = "weighted mean of raw scores over usable modalities (weights renormalised)"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": METHOD,
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
    }  # fmt: skip
    if not usable:
        notes.append("No modality produced usable evidence. This is NOT evidence that the media is authentic.")
        return FusionResult(FusionState.INSUFFICIENT_EVIDENCE, None, None, {}, {}, notes=tuple(notes), **common)

    raw = {e.modality: float(cfg.weights.get(e.modality, 0.0)) for e in usable}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("weights of usable modalities sum to zero")
    norm = {m: w / total for m, w in raw.items()}
    fused = sum(norm[e.modality] * (e.score or 0.0) for e in usable)
    flags = {e.modality.value: bool((e.score or 0.0) >= cfg.flag_threshold) for e in usable}
    state = FusionState.MULTIMODAL if len(usable) > 1 else FusionState.SINGLE_MODALITY
    if state is FusionState.SINGLE_MODALITY:
        notes.append("Result rests on one modality only; the other was not usable (see excluded_modalities). "
                     "An excluded modality says nothing about authenticity.")  # fmt: skip
    elif len(set(flags.values())) > 1:
        notes.append("The audio and visual branches disagree at the default threshold; inspect them separately.")
    if not cfg.weights_validated:
        notes.append("Fusion weights are an unvalidated equal-weight default.")
    if not cfg.threshold_validated:
        notes.append("The threshold is an unvalidated default, not an operating point chosen on validation data.")
    return FusionResult(state, fused, fused >= cfg.flag_threshold,
                        {m.value: w for m, w in norm.items()}, flags, notes=tuple(notes), **common)  # fmt: skip
