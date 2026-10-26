"""Contract tests for the evidence types (NOT fusion tests).

They verify that a missing/corrupted/unreliable modality can never be mistaken
for evidence of authenticity. Fusion-level behaviour with these same four
scenarios is a Phase 5 acceptance criterion and is not tested here.
"""

from __future__ import annotations

import pytest

from avdd.common.types import (
    Modality,
    ModalityEvidence,
    ModalityStatus,
    has_sufficient_evidence,
    usable_modalities,
)

A, V = Modality.AUDIO, Modality.VISUAL


def available(m: Modality, score: float = 0.2) -> ModalityEvidence:
    return ModalityEvidence(m, ModalityStatus.AVAILABLE, score=score, reliability=0.9)


def missing(m: Modality) -> ModalityEvidence:
    return ModalityEvidence(m, ModalityStatus.MISSING)


def test_missing_audio_leaves_only_visual_usable() -> None:
    ev = [missing(A), available(V)]
    assert usable_modalities(ev) == [V]
    assert has_sufficient_evidence(ev)
    assert not has_sufficient_evidence(ev, min_usable_modalities=2)


def test_missing_video_leaves_only_audio_usable() -> None:
    ev = [available(A), missing(V)]
    assert usable_modalities(ev) == [A]
    assert has_sufficient_evidence(ev)


def test_both_missing_is_insufficient_evidence_not_authentic() -> None:
    ev = [missing(A), missing(V)]
    assert usable_modalities(ev) == []
    assert not has_sufficient_evidence(ev)
    assert all(e.score is None for e in ev)  # no hidden "0.0 == real"


def test_unreliable_evidence_is_excluded_even_if_score_kept_for_display() -> None:
    unreliable = ModalityEvidence(A, ModalityStatus.UNRELIABLE, score=0.01, reasons=("silent audio",))
    ev = [unreliable, missing(V)]
    assert not unreliable.usable
    assert usable_modalities(ev) == []
    assert not has_sufficient_evidence(ev)


@pytest.mark.parametrize(
    "status", [ModalityStatus.MISSING, ModalityStatus.CORRUPTED, ModalityStatus.NOT_ANALYZED]
)
def test_absent_modality_cannot_carry_a_score(status: ModalityStatus) -> None:
    with pytest.raises(ValueError, match="not evidence of authenticity"):
        ModalityEvidence(A, status, score=0.0, reasons=("x",))


def test_available_requires_score() -> None:
    with pytest.raises(ValueError, match="requires a score"):
        ModalityEvidence(V, ModalityStatus.AVAILABLE)


def test_corrupted_and_unreliable_require_reasons() -> None:
    with pytest.raises(ValueError, match="requires at least one reason"):
        ModalityEvidence(A, ModalityStatus.CORRUPTED)
    with pytest.raises(ValueError, match="requires at least one reason"):
        ModalityEvidence(A, ModalityStatus.NOT_ANALYZED)
    assert not ModalityEvidence(A, ModalityStatus.NOT_ANALYZED, reasons=("no model",)).usable
    with pytest.raises(ValueError, match="requires at least one reason"):
        ModalityEvidence(A, ModalityStatus.UNRELIABLE, score=0.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_scores_must_be_in_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="within"):
        ModalityEvidence(A, ModalityStatus.AVAILABLE, score=bad)


def test_min_usable_modalities_must_be_positive() -> None:
    with pytest.raises(ValueError):
        has_sufficient_evidence([], min_usable_modalities=0)
