"""Audio preprocessing and branch tests (synthetic media + stub detectors: software tests only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avdd.audio.branch import run_audio_branch
from avdd.audio.extract import SAMPLE_RATE, assess_audio, decode_audio
from avdd.audio.windows import WINDOW_SAMPLES, make_windows
from avdd.common.errors import AudioError
from avdd.common.types import ModalityStatus
from tests.conftest import requires_ffmpeg
from tests.stubs import StubAudio

pytestmark = requires_ffmpeg


def test_decode_shape_dtype_range(media_dir: Path) -> None:
    w = decode_audio(media_dir / "video_audio.mp4")
    assert w.dtype == np.float32 and abs(len(w) / SAMPLE_RATE - 2.0) < 0.3
    assert float(np.abs(w).max()) <= 1.0


def test_decode_truncates_to_max_seconds(media_dir: Path) -> None:
    assert len(decode_audio(media_dir / "video_audio.mp4", max_seconds=1.0)) <= SAMPLE_RATE + 1


@pytest.mark.parametrize("name", ["video_only.mp4", "garbage.mp4", "truncated.mp4"])
def test_decode_failures_are_typed(media_dir: Path, name: str) -> None:
    with pytest.raises(AudioError) as ei:
        decode_audio(media_dir / name)
    assert ei.value.code == "decode_failed" and str(media_dir) not in ei.value.message


def test_usability_gate() -> None:
    sine = (0.1 * np.sin(2 * np.pi * 300 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)).astype(np.float32)
    assert assess_audio(sine)[0] == []
    assert "near-silent audio" in assess_audio(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))[0]
    assert any("shorter" in r for r in assess_audio(sine[:4000])[0])


def test_windowing() -> None:
    mk = lambda n: np.ones(n, dtype=np.float32)
    one = make_windows(mk(SAMPLE_RATE))  # 1 s: tiled to a full window
    assert len(one) == 1 and one[0].samples.shape == (WINDOW_SAMPLES,) and one[0].end_s == pytest.approx(1.0)
    ten = make_windows(mk(10 * SAMPLE_RATE))
    assert [w.start_s for w in ten] == pytest.approx([0.0, 4.0375, 8.075])
    assert len(make_windows(mk(WINDOW_SAMPLES + 8000))) == 1  # tail < 1 s dropped
    assert make_windows(mk(0)) == []


def test_branch_available_with_segments(face_dir: Path) -> None:
    det = StubAudio(0.7)
    out = run_audio_branch(face_dir / "face_audio.mp4", det)
    assert out.evidence.status is ModalityStatus.AVAILABLE and out.evidence.score == pytest.approx(0.7)
    seg = out.details["segments"]
    assert len(seg) == 1 and seg[0]["start_s"] == 0.0 and seg[0]["score"] == 0.7
    assert out.details["model"]["is_test_fixture"] is True


def test_branch_silent_audio_is_unreliable_and_detector_not_called(face_dir: Path) -> None:
    det = StubAudio()
    out = run_audio_branch(face_dir / "face_silent.mp4", det)
    assert out.evidence.status is ModalityStatus.UNRELIABLE and out.evidence.score is None
    assert det.calls == 0


@pytest.mark.parametrize("mode", ["raise", "nan"])
def test_branch_detector_failure_is_not_analyzed(face_dir: Path, mode: str) -> None:
    out = run_audio_branch(face_dir / "face_audio.mp4", StubAudio(mode=mode))
    assert out.evidence.status is ModalityStatus.NOT_ANALYZED and out.evidence.score is None
    assert "secret" not in str(out.evidence.reasons)


def test_branch_corrupted(media_dir: Path) -> None:
    out = run_audio_branch(media_dir / "garbage.mp4", StubAudio())
    assert out.evidence.status is ModalityStatus.CORRUPTED


@pytest.mark.parametrize("variant", ["AASIST-L", "AASIST"])
def test_real_aasist_weights_load_and_run(variant: str) -> None:
    """Software smoke test of the vendored pretrained weights. NOT a measure of detection accuracy."""
    pytest.importorskip("torch")
    from avdd.audio.aasist_detector import AasistDetector

    det = AasistDetector(variant)
    det.load()
    x = (np.random.default_rng(0).standard_normal(WINDOW_SAMPLES) * 0.05).astype(np.float32)
    s1, s2 = det.score_windows([x])[0], det.score_windows([x])[0]
    assert 0.0 <= s1 <= 1.0 and s1 == s2
    assert det.model_card.license.startswith("MIT") and "ASVspoof" in det.model_card.trained_on
