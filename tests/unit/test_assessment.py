"""Verdict rules. Pure logic: no models, no media."""

from __future__ import annotations

from avdd.common.types import Modality, ModalityEvidence, ModalityStatus
from avdd.fusion.assessment import Thresholds, build_assessment

A, V = Modality.AUDIO, Modality.VISUAL
TH = {A: Thresholds(0.2, 0.8), V: Thresholds(0.3, 0.6)}


def ok(m: Modality, s: float) -> ModalityEvidence:
    return ModalityEvidence(m, ModalityStatus.AVAILABLE, score=s)


def gone(m: Modality) -> ModalityEvidence:
    return ModalityEvidence(m, ModalityStatus.MISSING)


def verdict(*ev: ModalityEvidence) -> str:
    return str(build_assessment(ev, TH)["verdict"])


def test_verdicts() -> None:
    assert verdict(ok(A, 0.1), ok(V, 0.2)) == "likely_authentic"
    assert verdict(ok(A, 0.1), ok(V, 0.65)) == "likely_manipulated"  # one branch is enough
    assert verdict(ok(A, 0.5), ok(V, 0.1)) == "inconclusive"
    assert verdict(ok(A, 0.7), gone(V)) == "inconclusive"  # per-branch thresholds, not a global 0.5
    assert verdict(gone(A), gone(V)) == "insufficient_evidence"


def test_basis_and_score() -> None:
    r = build_assessment([ok(A, 0.1), gone(V)], TH)
    assert r["basis"] == "audio_only" and r["score"] == 0.1 and r["branch_verdicts"]["visual"] is None
    r = build_assessment([gone(A), gone(V)], TH)
    assert r["basis"] == "none" and r["score"] is None and "says nothing about authenticity" in r["summary"]
