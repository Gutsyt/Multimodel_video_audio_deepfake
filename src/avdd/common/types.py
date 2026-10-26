"""Shared evidence contract: availability of a modality is NOT evidence of authenticity.

A missing, corrupted, or unreliable modality must never be encoded as a "real"
score (e.g. 0.0). This module makes that mistake impossible to construct:
``ModalityEvidence`` rejects a score for MISSING/CORRUPTED modalities, and
``usable_modalities`` only returns modalities whose status is AVAILABLE.

Scores here are raw model outputs in [0, 1] interpreted as "evidence of
manipulation". They are not calibrated probabilities unless a calibration step
(evaluated on validation data) says so.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    """The two modalities analysed by the system."""

    AUDIO = "audio"
    VISUAL = "visual"


class ModalityStatus(str, Enum):
    """Whether a modality can contribute evidence.

    AVAILABLE   - analysed successfully; ``score`` is meaningful.
    MISSING     - the stream does not exist (e.g. video without an audio track).
    CORRUPTED   - the stream/container exists but could not be decoded.
    NOT_ANALYZED - the stream exists but no analysis was run (e.g. model not
                  loaded, inference failed). A system limitation, not a media property.
    UNRELIABLE  - analysed, but quality checks (silence, no face, tiny face, ...)
                  say the score should not be trusted; excluded from decisions.
    """

    AVAILABLE = "available"
    MISSING = "missing"
    CORRUPTED = "corrupted"
    NOT_ANALYZED = "not_analyzed"
    UNRELIABLE = "unreliable"


_ABSENT = (ModalityStatus.MISSING, ModalityStatus.CORRUPTED, ModalityStatus.NOT_ANALYZED)


@dataclass(frozen=True)
class ModalityEvidence:
    """Evidence from one modality, with validated status/score consistency."""

    modality: Modality
    status: ModalityStatus
    score: float | None = None
    reliability: float | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (("score", self.score), ("reliability", self.reliability)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value!r}")

        if self.status is ModalityStatus.AVAILABLE:
            if self.score is None:
                raise ValueError("AVAILABLE evidence requires a score")
        elif self.status in _ABSENT:
            if self.score is not None or self.reliability is not None:
                raise ValueError(
                    f"{self.status.value} modality must not carry a score or reliability; "
                    "an absent modality is not evidence of authenticity"
                )
            if self.status is not ModalityStatus.MISSING and not self.reasons:
                raise ValueError(f"{self.status.value.upper()} evidence requires at least one reason")
        elif self.status is ModalityStatus.UNRELIABLE and not self.reasons:
            raise ValueError("UNRELIABLE evidence requires at least one reason")

    @property
    def usable(self) -> bool:
        """True only when the evidence may be used in a decision."""
        return self.status is ModalityStatus.AVAILABLE


def usable_modalities(evidence: Iterable[ModalityEvidence]) -> list[Modality]:
    """Return modalities whose evidence may be used (status AVAILABLE only)."""
    return [e.modality for e in evidence if e.usable]


def has_sufficient_evidence(
    evidence: Iterable[ModalityEvidence], min_usable_modalities: int = 1
) -> bool:
    """True if at least ``min_usable_modalities`` distinct modalities are usable.

    When this is False the system must report "insufficient evidence", never
    "authentic".
    """
    if min_usable_modalities < 1:
        raise ValueError("min_usable_modalities must be >= 1")
    return len(set(usable_modalities(evidence))) >= min_usable_modalities
