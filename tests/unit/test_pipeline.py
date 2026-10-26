"""End-to-end pipeline tests with STUB detectors on synthetic clips (software behaviour only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from avdd.common.errors import AvddError
from avdd.fusion.score_fusion import FusionState
from avdd.pipeline.analyze import analyze_media
from avdd.video.faces import HaarFaceDetector
from tests.conftest import requires_ffmpeg
from tests.stubs import StubAudio, StubFace

pytestmark = requires_ffmpeg


def run(path: Path, audio: StubAudio | None = None, face: StubFace | None = None) -> dict:  # type: ignore[type-arg]
    return analyze_media(path, audio, face, HaarFaceDetector())


def test_full_multimodal_result_shape_and_privacy(face_dir: Path) -> None:
    r = run(face_dir / "face_audio.mp4", StubAudio(0.8), StubFace(0.2))
    assert r["outcome"] == "analyzed" and r["is_test_fixture"] is True
    assert r["modalities"]["audio"]["status"] == r["modalities"]["visual"]["status"] == "available"
    assert r["fusion"]["state"] == FusionState.MULTIMODAL.value and r["fusion"]["fused_score"] == pytest.approx(0.5)
    assert r["interpretation"]["calibrated"] is False and r["fusion"]["calibrated"] is False
    text = json.dumps(r)
    assert str(face_dir) not in text and "face_audio" not in text


def test_video_without_audio_track(face_dir: Path) -> None:
    r = run(face_dir / "face_noaudio.mp4", StubAudio(0.9), StubFace(0.1))
    assert r["modalities"]["audio"]["status"] == "missing" and r["modalities"]["audio"]["score"] is None
    assert r["fusion"]["state"] == "single_modality" and r["fusion"]["fused_score"] == pytest.approx(0.1)


def test_audio_only_file_has_missing_video(media_dir: Path) -> None:
    r = run(media_dir / "audio_only.wav", StubAudio(0.6), StubFace())
    assert r["modalities"]["visual"]["status"] == "missing"
    assert r["fusion"]["state"] == "single_modality" and r["fusion"]["fused_score"] == pytest.approx(0.6)


@pytest.mark.parametrize("name", ["garbage.mp4", "truncated.mp4", "notes.txt"])
def test_unreadable_media_is_insufficient_evidence(media_dir: Path, name: str) -> None:
    r = run(media_dir / name, StubAudio(), StubFace())
    assert r["outcome"] == "unreadable_media"
    assert {m["status"] for m in r["modalities"].values()} == {"corrupted"}
    assert r["fusion"]["state"] == "insufficient_evidence" and r["fusion"]["fused_score"] is None


def test_unreliable_audio_leaves_visual_only(face_dir: Path) -> None:
    r = run(face_dir / "face_silent.mp4", StubAudio(0.99), StubFace(0.2))
    assert r["modalities"]["audio"]["status"] == "unreliable"
    assert r["fusion"]["fused_score"] == pytest.approx(0.2)


def test_no_models_loaded_is_not_analyzed_not_authentic(face_dir: Path) -> None:
    r = run(face_dir / "face_audio.mp4")
    assert {m["status"] for m in r["modalities"].values()} == {"not_analyzed"}
    assert r["fusion"]["state"] == "insufficient_evidence"


def test_missing_input_raises_structured_error(tmp_path: Path) -> None:
    with pytest.raises(AvddError) as ei:
        run(tmp_path / "nope.mp4")
    assert ei.value.code == "input_not_found"
