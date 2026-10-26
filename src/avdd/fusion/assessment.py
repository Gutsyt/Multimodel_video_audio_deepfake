"""The overall assessment: one explicit vocabulary on top of per-branch evidence and fusion.

``flagged``               a raw score reached the (unvalidated) default threshold.
``not_flagged``           no usable score reached it. This does NOT mean the media is authentic.
``insufficient_evidence`` no modality produced a usable score, so nothing can be said either way.

The basis says what the assessment rests on. With two usable branches it is the fused score
(``multimodal``). With one it is that branch's own score (``audio_only`` / ``visual_only``) and it
must never be presented as a multimodal result.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from avdd.common.types import Modality, ModalityEvidence
from avdd.fusion.score_fusion import FusionResult, FusionState


class Decision(str, Enum):
    FLAGGED = "flagged"
    NOT_FLAGGED = "not_flagged"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Basis(str, Enum):
    MULTIMODAL = "multimodal"
    AUDIO_ONLY = "audio_only"
    VISUAL_ONLY = "visual_only"
    NONE = "none"


NOT_AUTHENTICITY_NOTE = (
    "'Not flagged' only means no usable score reached the default threshold. It does not mean the "
    "media is authentic: the models are pretrained, unvalidated on this project's data, and can miss "
    "manipulations, while genuine media can be flagged."
)

_SUMMARY = {
    (Decision.FLAGGED, Basis.MULTIMODAL): "The fused raw score of the audio and visual branches is at or above the default threshold.",
    (Decision.NOT_FLAGGED, Basis.MULTIMODAL): "The fused raw score of the audio and visual branches is below the default threshold.",
    (Decision.FLAGGED, Basis.AUDIO_ONLY): "Audio-only result: the audio branch's raw score is at or above the default threshold. No visual evidence was usable and nothing was fused.",
    (Decision.NOT_FLAGGED, Basis.AUDIO_ONLY): "Audio-only result: the audio branch's raw score is below the default threshold. No visual evidence was usable and nothing was fused.",
    (Decision.FLAGGED, Basis.VISUAL_ONLY): "Visual-only result: the visual branch's raw score is at or above the default threshold. No audio evidence was usable and nothing was fused.",
    (Decision.NOT_FLAGGED, Basis.VISUAL_ONLY): "Visual-only result: the visual branch's raw score is below the default threshold. No audio evidence was usable and nothing was fused.",
    (Decision.INSUFFICIENT_EVIDENCE, Basis.NONE): "Neither the audio nor the visual branch produced usable evidence, so no assessment can be made. This is not evidence that the media is authentic.",
}  # fmt: skip


def build_assessment(evidence: Sequence[ModalityEvidence], fusion: FusionResult) -> dict[str, Any]:
    """Derive the overall assessment from branch evidence and the fusion result."""
    usable = {e.modality: e for e in evidence if e.usable}
    decisions: dict[str, str | None] = {
        e.modality.value: (
            None if not e.usable
            else (Decision.FLAGGED if (e.score or 0.0) >= fusion.threshold else Decision.NOT_FLAGGED).value
        )
        for e in evidence
    }  # fmt: skip
    score: float | None
    score_kind: str | None
    if fusion.state is FusionState.MULTIMODAL:
        basis, score, score_kind = Basis.MULTIMODAL, fusion.fused_score, "fused_raw_score"
        decision = Decision.FLAGGED if fusion.flagged else Decision.NOT_FLAGGED
    elif fusion.state is FusionState.SINGLE_MODALITY:
        (only,) = usable.values()
        basis = Basis.AUDIO_ONLY if only.modality is Modality.AUDIO else Basis.VISUAL_ONLY
        score, score_kind = only.score, f"{only.modality.value}_branch_raw_score"
        decision = Decision(decisions[only.modality.value])
    else:
        basis, score, score_kind, decision = Basis.NONE, None, None, Decision.INSUFFICIENT_EVIDENCE
    used_flags = {v for v in decisions.values() if v is not None}
    agree = (len(used_flags) == 1) if basis is Basis.MULTIMODAL else None
    summary = _SUMMARY[(decision, basis)]
    if agree is False:
        summary += (" The two branches disagree at the threshold, so this rests on an unvalidated average of "
                    "them; read each branch separately.")
    return {
        "status": decision.value,
        "basis": basis.value,
        "summary": summary,
        "branches_agree": agree,
        "score": None if score is None else round(score, 4),
        "score_kind": score_kind,
        "threshold": fusion.threshold,
        "threshold_validated": fusion.threshold_validated,
        "branch_decisions": decisions,
        "modalities_used": sorted(m.value for m in usable),
        "calibrated": False,
        "not_authenticity_note": NOT_AUTHENTICITY_NOTE,
    }
