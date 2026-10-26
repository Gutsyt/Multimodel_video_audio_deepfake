"""Video preprocessing and branch tests (synthetic media + stubs: software tests only)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")

from avdd.common.errors import VideoError
from avdd.common.types import ModalityStatus
from avdd.video.branch import run_video_branch
from avdd.video.faces import HaarFaceDetector, crop_face_rgb
from avdd.video.frames import sample_frames
from tests.conftest import requires_ffmpeg
from tests.stubs import StubBoxes, StubFace

pytestmark = requires_ffmpeg


def test_sample_frames_uniform_and_ordered(face_dir: Path) -> None:
    fr = sample_frames(face_dir / "face_audio.mp4", n_frames=8)
    assert 2 <= len(fr) <= 8
    assert [f.t_s for f in fr] == sorted(f.t_s for f in fr) and fr[-1].t_s <= 3.0


def test_sample_frames_corrupt_raises(media_dir: Path) -> None:
    with pytest.raises(VideoError) as ei:
        sample_frames(media_dir / "garbage.mp4")
    assert str(media_dir) not in ei.value.message


def test_haar_finds_the_public_domain_test_face_and_not_a_test_pattern(face_dir: Path, media_dir: Path) -> None:
    det = HaarFaceDetector()
    face_frame = sample_frames(face_dir / "face_noaudio.mp4", 1)[0].image_bgr
    boxes = det.detect(face_frame)
    assert len(boxes) >= 1
    crop = crop_face_rgb(face_frame, boxes[0])
    assert crop.ndim == 3 and crop.shape[2] == 3 and min(crop.shape[:2]) > 40
    assert det.detect(sample_frames(media_dir / "video_only.mp4", 1)[0].image_bgr) == []


def test_branch_available_with_per_frame_timestamps(face_dir: Path) -> None:
    det = StubFace(0.3)
    out = run_video_branch(face_dir / "face_noaudio.mp4", det, HaarFaceDetector())
    assert out.evidence.status is ModalityStatus.AVAILABLE and out.evidence.score == pytest.approx(0.3)
    frames = out.details["frames"]
    assert all("t_s" in f and "n_faces" in f for f in frames)
    assert all(f["score"] == 0.3 for f in frames if f["n_faces"])
    assert out.details["face_detector"] == "opencv-haar-frontalface-default"


def test_branch_no_face_is_unreliable_not_real(media_dir: Path) -> None:
    det = StubFace()
    out = run_video_branch(media_dir / "video_only.mp4", det, HaarFaceDetector())
    assert out.evidence.status is ModalityStatus.UNRELIABLE and out.evidence.score is None
    assert det.calls == 0 and "faces found in 0 of" in out.evidence.reasons[0]


def test_branch_multiple_faces_adds_caveat(face_dir: Path) -> None:
    out = run_video_branch(face_dir / "face_noaudio.mp4", StubFace(0.5), StubBoxes(2))
    assert out.evidence.status is ModalityStatus.AVAILABLE
    assert any("multiple faces" in r for r in out.evidence.reasons)


def test_branch_failures(face_dir: Path, media_dir: Path) -> None:
    out = run_video_branch(face_dir / "face_noaudio.mp4", StubFace(mode="raise"), StubBoxes())
    assert out.evidence.status is ModalityStatus.NOT_ANALYZED
    assert run_video_branch(media_dir / "garbage.mp4", StubFace(), StubBoxes()).evidence.status is ModalityStatus.CORRUPTED
