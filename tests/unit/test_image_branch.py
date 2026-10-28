"""Image-only visual path with STUB detectors on synthetic images (software behaviour only)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from avdd.common.errors import VideoError
from avdd.pipeline.analyze import analyze_media
from avdd.video.branch import run_image_branch
from avdd.video.faces import YuNetFaceDetector
from avdd.video.frames import load_image_frame
from tests.conftest import requires_ffmpeg
from tests.stubs import StubAudio, StubBoxes, StubFace

pytestmark = requires_ffmpeg


def test_image_with_a_face_is_analysed_from_one_frame(kinds_dir: Path) -> None:
    out = run_image_branch(kinds_dir / "face.png", StubFace(0.4), StubBoxes())
    assert out.evidence.usable and out.evidence.score == pytest.approx(0.4)
    assert out.details["source"] == "image" and out.details["n_frames_sampled"] == 1
    frame = out.details["frames"][0]
    assert frame["t_s"] is None and frame["face_area_ratio"] > 0 and frame["crop_px"][0] > 0
    assert out.details["timestamps"] == "none (still image)"


def test_image_without_a_face_is_unreliable_not_real(kinds_dir: Path) -> None:
    det = StubFace(0.0)
    out = run_image_branch(kinds_dir / "noface.jpg", det, YuNetFaceDetector())
    assert out.evidence.score is None and not out.evidence.usable
    assert det.calls == 0
    assert out.evidence.reasons == ("no face was detected in the image",)


@pytest.mark.parametrize("name", ["fake.jpg", "truncated.jpg"])
def test_undecodable_image_is_corrupted(kinds_dir: Path, name: str) -> None:
    out = run_image_branch(kinds_dir / name, StubFace(), StubBoxes())
    assert out.evidence.status.value == "corrupted" and out.details["error_code"] == "image_decode_failed"


def test_pixel_cap_guards_decompression_bombs(kinds_dir: Path) -> None:
    with pytest.raises(VideoError) as ei:
        load_image_frame(kinds_dir / "face.png", max_pixels=1000)
    assert ei.value.code == "image_too_large"


@pytest.mark.parametrize("name", ["face.jpg", "face.png", "face.webp", "face.bmp"])
def test_pipeline_runs_the_visual_branch_only_for_supported_images(kinds_dir: Path, name: str) -> None:
    r = analyze_media(kinds_dir / name, StubAudio(0.9), StubFace(0.3), YuNetFaceDetector())
    assert r["media"]["kind"] == "image"
    assert r["modalities"]["audio"]["status"] == "missing" and r["modalities"]["audio"]["score"] is None
    assert r["modalities"]["audio"]["reasons"] == ["this image has no audio"]
    assert r["modalities"]["visual"]["source"] == "image"
    assert r["assessment"]["basis"] == "visual_only"


def test_pipeline_image_without_face_is_insufficient_evidence(kinds_dir: Path) -> None:
    r = analyze_media(kinds_dir / "noface.jpg", StubAudio(0.9), StubFace(0.9), YuNetFaceDetector())
    assert r["assessment"]["verdict"] == "insufficient_evidence" and r["assessment"]["score"] is None
