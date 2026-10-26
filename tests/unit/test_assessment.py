"""Overall assessment vocabulary and basis. Pure logic: no models, no media."""

from __future__ import annotations

import pytest

from avdd.common.types import Modality, ModalityEvidence, ModalityStatus
from avdd.fusion.assessment import NOT_AUTHENTICITY_NOTE, build_assessment
from avdd.fusion.score_fusion import FusionConfig, fuse_scores

A, V = Modality.AUDIO, Modality.VISUAL


def ok(m: Modality, s: float) -> ModalityEvidence:
    return ModalityEvidence(m, ModalityStatus.AVAILABLE, score=s)


def gone(m: Modality, st: ModalityStatus = ModalityStatus.MISSING) -> ModalityEvidence:
    return ModalityEvidence(m, st, reasons=("because",))


def assess(*evidence: ModalityEvidence, cfg: FusionConfig | None = None) -> dict:  # type: ignore[type-arg]
    return build_assessment(evidence, fuse_scores(evidence, cfg))


def test_multimodal_flagged_uses_the_fused_score() -> None:
    a = assess(ok(A, 1.0), ok(V, 0.76))
    assert (a["status"], a["basis"], a["score_kind"]) == ("flagged", "multimodal", "fused_raw_score")
    assert a["score"] == pytest.approx(0.88)
    assert a["branch_decisions"] == {"audio": "flagged", "visual": "flagged"}
    assert a["threshold_validated"] is False and a["calibrated"] is False


def test_multimodal_not_flagged_is_not_authentic() -> None:
    a = assess(ok(A, 0.1), ok(V, 0.2))
    assert a["status"] == "not_flagged" and a["basis"] == "multimodal"
    assert "does not mean the media is authentic" in a["not_authenticity_note"]
    assert a["not_authenticity_note"] == NOT_AUTHENTICITY_NOTE


def test_single_audio_branch_is_not_described_as_multimodal() -> None:
    a = assess(ok(A, 0.9), gone(V))
    assert (a["status"], a["basis"], a["score_kind"]) == ("flagged", "audio_only", "audio_branch_raw_score")
    assert a["score"] == pytest.approx(0.9)  # the branch's own score, not an average with a default
    assert a["branch_decisions"] == {"audio": "flagged", "visual": None}
    assert a["modalities_used"] == ["audio"]
    assert "nothing was fused" in a["summary"] and "multimodal" not in a["summary"].lower()


def test_single_visual_branch_not_flagged() -> None:
    a = assess(gone(A), ok(V, 0.3))
    assert (a["status"], a["basis"]) == ("not_flagged", "visual_only")
    assert a["branch_decisions"] == {"audio": None, "visual": "not_flagged"}


@pytest.mark.parametrize("st", [ModalityStatus.MISSING, ModalityStatus.CORRUPTED, ModalityStatus.NOT_ANALYZED])
def test_nothing_usable_is_insufficient_evidence(st: ModalityStatus) -> None:
    a = assess(gone(A, st), gone(V, st))
    assert (a["status"], a["basis"]) == ("insufficient_evidence", "none")
    assert a["score"] is None and a["score_kind"] is None
    assert a["branch_decisions"] == {"audio": None, "visual": None}
    assert "not evidence that the media is authentic" in a["summary"]


def test_unreliable_branch_never_decides() -> None:
    unreliable = ModalityEvidence(A, ModalityStatus.UNRELIABLE, score=0.99, reasons=("silent",))
    a = assess(unreliable, gone(V))
    assert a["status"] == "insufficient_evidence" and a["branch_decisions"]["audio"] is None


def test_threshold_is_the_configured_one_and_reported() -> None:
    cfg = FusionConfig(flag_threshold=0.95)
    a = assess(ok(A, 0.9), gone(V), cfg=cfg)
    assert a["status"] == "not_flagged" and a["threshold"] == 0.95


def test_score_exactly_at_threshold_is_flagged() -> None:
    assert assess(ok(A, 0.5), gone(V))["status"] == "flagged"
