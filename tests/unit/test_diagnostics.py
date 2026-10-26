"""Branch diagnostics with detectors that expose logits (TEST DOUBLES, not real models)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("cv2")

from avdd.audio.branch import run_audio_branch
from avdd.common.protocols import ScoredItem
from avdd.common.types import Modality
from avdd.video.branch import run_image_branch
from tests.conftest import requires_ffmpeg
from tests.stubs import StubAudio, StubBoxes, StubFace, _card

pytestmark = requires_ffmpeg


class DetailedAudio(StubAudio):
    """Like AASIST's interface: two logits per window, class order (spoof, bona_fide)."""

    output_semantics: ClassVar[dict[str, Any]] = {
        "class_labels": ["spoof", "bona_fide"], "score_class": "spoof", "score_class_index": 0}
    device = "cpu"

    def score_windows_detailed(self, windows: Sequence[NDArray[np.float32]]) -> list[ScoredItem]:
        return [ScoredItem(self.score, {"spoof": 9.0, "bona_fide": -9.0},
                           {"spoof": self.score, "bona_fide": 1 - self.score}) for _ in windows]


class DetailedFace(StubFace):
    output_semantics: ClassVar[dict[str, Any]] = {
        "class_labels": ["Realism", "Deepfake"], "score_class": "Deepfake", "score_class_index": 1}

    def score_faces_detailed(self, faces: Sequence[NDArray[np.uint8]]) -> list[ScoredItem]:
        return [ScoredItem(self.score, {"Realism": -1.0, "Deepfake": 1.0},
                           {"Realism": 1 - self.score, "Deepfake": self.score}) for _ in faces]


def test_audio_branch_reports_logits_stats_and_semantics(kinds_dir: Path) -> None:
    out = run_audio_branch(kinds_dir / "tone.flac", DetailedAudio(0.995))
    d = out.details
    seg = d["segments"][0]
    assert seg["logits"] == {"spoof": 9.0, "bona_fide": -9.0} and seg["class_probs"]["bona_fide"] == pytest.approx(0.005, abs=1e-3)
    w = d["diagnostics"]["windows"]
    assert w["n_saturated"] == w["n_windows"] == 1 and w["mean"] == w["max"] == pytest.approx(0.995, abs=1e-4)
    assert d["diagnostics"]["output_semantics"]["score_class_index"] == 0
    inp = d["diagnostics"]["input"]
    assert inp["target_sample_rate"] == 16000 and inp["device"] == "cpu" and inp["window_samples"] == 64600
    assert inp["peak_abs"] > 0 and inp["clipping_ratio"] == 0.0 and "none" in inp["normalization"]
    assert out.evidence.score == pytest.approx(0.995)  # the reported score is still the plain mean


def test_audio_branch_works_with_detectors_that_only_return_scores(kinds_dir: Path) -> None:
    out = run_audio_branch(kinds_dir / "tone.flac", StubAudio(0.3))
    assert out.details["segments"][0]["logits"] is None and out.evidence.score == pytest.approx(0.3)


def test_unreliable_audio_still_explains_the_input(tmp_path: Path) -> None:
    import subprocess

    quiet = tmp_path / "quiet.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "3", str(quiet)], check=True, timeout=60)  # fmt: skip
    out = run_audio_branch(quiet, StubAudio())
    assert out.evidence.status.value == "unreliable"
    assert out.details["diagnostics"]["input"]["peak_abs"] == 0.0


def test_visual_branch_reports_labels_and_per_frame_probabilities(kinds_dir: Path) -> None:
    out = run_image_branch(kinds_dir / "face.png", DetailedFace(0.76), StubBoxes())
    fr = out.details["frames"][0]
    assert fr["class_probs"] == {"Realism": 0.24, "Deepfake": 0.76} and fr["logits"]["Deepfake"] == 1.0
    d = out.details["diagnostics"]
    assert d["output_semantics"]["score_class"] == "Deepfake" and d["scores"]["n_at_or_above_threshold"] == 1
    assert d["input"]["face_detector"] == "stub-boxes" and d["input"]["frame_width"] > 0


def test_stub_card_is_flagged_as_test_fixture() -> None:
    assert _card(Modality.AUDIO).is_test_fixture


class Disagreeing(DetailedAudio):
    def score_windows_detailed(self, windows: Sequence[NDArray[np.float32]]) -> list[ScoredItem]:
        vals = [0.01, 0.99, 0.99][: len(windows)]
        return [ScoredItem(v, {"spoof": 0.0, "bona_fide": 0.0}, {"spoof": v, "bona_fide": 1 - v}) for v in vals]


def test_strongly_disagreeing_windows_are_called_out_without_changing_the_score(tmp_path: Path) -> None:
    import subprocess

    wav = tmp_path / "long.wav"  # 12 s -> 3 windows
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=12:sample_rate=16000",
                    str(wav)], check=True, timeout=60)  # fmt: skip
    out = run_audio_branch(wav, Disagreeing(0.5))
    assert out.evidence.score == pytest.approx((0.01 + 0.99 + 0.99) / 3)  # plain mean, untouched
    assert out.details["diagnostics"]["windows"]["range"] == pytest.approx(0.98)
    assert any("disagree strongly" in r for r in out.evidence.reasons)
