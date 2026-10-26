"""Fusion A tests, including the four missing-evidence scenarios. Contract-level, no models."""

from __future__ import annotations

import pytest

from avdd.common.types import Modality, ModalityEvidence, ModalityStatus
from avdd.fusion.score_fusion import FusionConfig, FusionState, fuse_scores

A, V = Modality.AUDIO, Modality.VISUAL


def ok(m: Modality, s: float) -> ModalityEvidence:
    return ModalityEvidence(m, ModalityStatus.AVAILABLE, score=s)


def absent(m: Modality, st: ModalityStatus) -> ModalityEvidence:
    return ModalityEvidence(m, st, reasons=() if st is ModalityStatus.MISSING else ("x",))


def test_both_usable_equal_weights() -> None:
    r = fuse_scores([ok(A, 0.8), ok(V, 0.2)])
    assert r.state is FusionState.MULTIMODAL
    assert r.fused_score == pytest.approx(0.5) and r.flagged is True
    assert r.weights_used == {"audio": 0.5, "visual": 0.5}
    assert any("disagree" in n for n in r.notes)


def test_missing_audio_uses_visual_only_and_does_not_average_in_zero() -> None:
    r = fuse_scores([absent(A, ModalityStatus.MISSING), ok(V, 0.1)])
    assert r.state is FusionState.SINGLE_MODALITY
    # No fusion happens with one branch: there is no combined score at all (and certainly not 0.05).
    assert r.fused_score is None and r.flagged is None and not r.fusion_performed
    assert r.branch_flags == {"visual": False}
    assert r.excluded == {"audio": "missing"} and r.weights_used == {}
    assert any("says nothing about authenticity" in n for n in r.notes)


def test_missing_video_uses_audio_only() -> None:
    r = fuse_scores([ok(A, 0.9), absent(V, ModalityStatus.MISSING)])
    assert r.state is FusionState.SINGLE_MODALITY and r.fused_score is None
    assert r.branch_flags == {"audio": True}


def test_both_missing_is_insufficient_evidence_never_authentic() -> None:
    r = fuse_scores([absent(A, ModalityStatus.MISSING), absent(V, ModalityStatus.MISSING)])
    assert r.state is FusionState.INSUFFICIENT_EVIDENCE
    assert r.fused_score is None and r.flagged is None
    assert any("NOT evidence that the media is authentic" in n for n in r.notes)


@pytest.mark.parametrize("st", [ModalityStatus.CORRUPTED, ModalityStatus.NOT_ANALYZED])
def test_other_absent_statuses_are_excluded(st: ModalityStatus) -> None:
    r = fuse_scores([absent(A, st), ok(V, 0.4)])
    assert r.excluded == {"audio": st.value} and r.state is FusionState.SINGLE_MODALITY


def test_unreliable_evidence_is_excluded_even_with_a_score() -> None:
    unreliable = ModalityEvidence(A, ModalityStatus.UNRELIABLE, score=0.99, reasons=("silent",))
    r = fuse_scores([unreliable, ok(V, 0.2)])
    assert r.fused_score is None and r.branch_flags == {"visual": False}
    assert r.excluded == {"audio": "unreliable"}
    both = fuse_scores([unreliable, absent(V, ModalityStatus.MISSING)])
    assert both.state is FusionState.INSUFFICIENT_EVIDENCE


def test_custom_weights_are_renormalised_over_usable_modalities() -> None:
    cfg = FusionConfig(weights={A: 0.2, V: 0.8})
    assert fuse_scores([ok(A, 1.0), ok(V, 0.0)], cfg).fused_score == pytest.approx(0.2)
    assert fuse_scores([ok(A, 0.6), absent(V, ModalityStatus.MISSING)], cfg).weights_used == {}


def test_result_is_labelled_uncalibrated_and_unvalidated() -> None:
    d = fuse_scores([ok(A, 0.5), ok(V, 0.5)]).to_dict()
    assert d["calibrated"] is False and d["weights_validated"] is False and d["threshold_validated"] is False
    assert any("unvalidated" in n for n in d["notes"])


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        fuse_scores([ok(A, 0.1), ok(A, 0.2)])
    with pytest.raises(ValueError, match="non-negative"):
        FusionConfig(weights={A: -1.0, V: 1.0})
    with pytest.raises(ValueError, match="sum to zero"):
        fuse_scores([ok(A, 0.1), ok(V, 0.2)], FusionConfig(weights={A: 0.0, V: 0.0}))
    with pytest.raises(ValueError):
        FusionConfig(flag_threshold=1.5)
