"""Overall verdict from calibrated per-branch scores and their validated thresholds.

Each branch score is a calibrated P(fake). Each branch has two thresholds chosen on validation
data (``scripts/train_heads.py``): ``t_fake`` (at most ~2% of genuine media reach it) and
``t_real`` (about 90% of genuine media fall at or below it).

  likely_manipulated     any usable branch >= its t_fake
  likely_authentic       every usable branch <= its t_real
  inconclusive           otherwise (in between, or the branches disagree)
  insufficient_evidence  no usable branch

Flagging on "any branch" keeps one-sided fakes (voice clone over real video, face swap over real
voice) detectable; the low per-branch false-alarm rate keeps genuine media from being flagged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from avdd.common.types import Modality, ModalityEvidence


@dataclass(frozen=True)
class Thresholds:
    t_real: float = 0.3
    t_fake: float = 0.7

    def verdict(self, score: float) -> str:
        if score >= self.t_fake:
            return "likely_manipulated"
        return "likely_authentic" if score <= self.t_real else "inconclusive"


def thresholds_of(detector: object) -> Thresholds:
    """Thresholds from a detector's trained head; defaults for detectors without one (test doubles)."""
    head = getattr(detector, "head", None)
    return Thresholds(head.t_real, head.t_fake) if head is not None else Thresholds()


_SUMMARY = {
    "likely_manipulated": "Signs of manipulation were found",
    "likely_authentic": "No signs of manipulation were found",
    "inconclusive": "The evidence is mixed or borderline",
    "insufficient_evidence": "Nothing could be analysed",
}
_WHAT = {Modality.AUDIO: "voice", Modality.VISUAL: "face"}


def build_assessment(evidence: Sequence[ModalityEvidence], thresholds: Mapping[Modality, Thresholds]) -> dict[str, Any]:
    usable = [e for e in evidence if e.usable]
    branch = {e.modality.value: thresholds[e.modality].verdict(e.score or 0.0) if e.usable else None for e in evidence}
    used = [v for v in branch.values() if v is not None]
    if not used:
        verdict = "insufficient_evidence"
    elif "likely_manipulated" in used:
        verdict = "likely_manipulated"
    elif all(v == "likely_authentic" for v in used):
        verdict = "likely_authentic"
    else:
        verdict = "inconclusive"
    basis = ("multimodal" if len(usable) == 2 else
             f"{usable[0].modality.value}_only" if usable else "none")  # fmt: skip
    why = [f"the {_WHAT[e.modality]} looks {'manipulated' if branch[e.modality.value] == 'likely_manipulated' else 'genuine' if branch[e.modality.value] == 'likely_authentic' else 'borderline'} "
           f"({round(100 * (e.score or 0))}% fake probability)" for e in usable]  # fmt: skip
    summary = _SUMMARY[verdict] + (": " + "; ".join(why) + "." if why else
                                   ". No face or speech was usable, so this says nothing about authenticity.")  # fmt: skip
    return {
        "verdict": verdict,
        "basis": basis,
        "summary": summary,
        "score": max((e.score or 0.0 for e in usable), default=None),
        "branch_verdicts": branch,
        "thresholds": {m.value: {"t_real": round(t.t_real, 4), "t_fake": round(t.t_fake, 4)} for m, t in thresholds.items()},
        "modalities_used": sorted(e.modality.value for e in usable),
    }
