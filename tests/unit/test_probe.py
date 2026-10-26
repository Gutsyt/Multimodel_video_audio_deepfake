"""ffprobe wrapper and availability mapping on SYNTHETIC fixtures (software tests only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from avdd.common.types import Modality, ModalityStatus
from avdd.media.availability import availability_from_probe
from avdd.media.probe import MediaProbe, ProbeStatus, probe_media
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def test_video_with_audio(media_dir: Path) -> None:
    p = probe_media(media_dir / "video_audio.mp4")
    assert p.status is ProbeStatus.OK
    assert p.has_video and p.has_audio
    assert (p.width, p.height) == (160, 120)
    assert p.fps == pytest.approx(25.0)
    assert p.sample_rate == 16000
    assert p.duration_s == pytest.approx(2.0, abs=0.3)
    assert availability_from_probe(p) == {
        Modality.AUDIO: ModalityStatus.AVAILABLE,
        Modality.VISUAL: ModalityStatus.AVAILABLE,
    }


def test_video_without_audio_is_missing_audio_not_corrupted(media_dir: Path) -> None:
    p = probe_media(media_dir / "video_only.mp4")
    assert p.status is ProbeStatus.OK and p.has_video and not p.has_audio
    avail = availability_from_probe(p)
    assert avail[Modality.AUDIO] is ModalityStatus.MISSING
    assert avail[Modality.VISUAL] is ModalityStatus.AVAILABLE


def test_audio_only_is_missing_video(media_dir: Path) -> None:
    p = probe_media(media_dir / "audio_only.wav")
    assert p.status is ProbeStatus.OK and p.has_audio and not p.has_video
    avail = availability_from_probe(p)
    assert avail[Modality.VISUAL] is ModalityStatus.MISSING
    assert avail[Modality.AUDIO] is ModalityStatus.AVAILABLE


@pytest.mark.parametrize("name", ["garbage.mp4", "truncated.mp4", "notes.txt"])
def test_unparseable_files_fail_cleanly_without_leaking_paths(media_dir: Path, name: str) -> None:
    p = probe_media(media_dir / name)
    assert p.status is ProbeStatus.FAILED
    assert p.error is not None and str(media_dir) not in p.error
    assert set(availability_from_probe(p).values()) == {ModalityStatus.CORRUPTED}


def test_extension_is_not_trusted(media_dir: Path, tmp_path: Path) -> None:
    disguised = tmp_path / "actually_audio.mp4"
    disguised.write_bytes((media_dir / "audio_only.wav").read_bytes())
    p = probe_media(disguised)
    assert p.status is ProbeStatus.OK and p.has_audio and not p.has_video


def test_missing_file_and_missing_tool_are_not_reported_as_corruption(media_dir: Path) -> None:
    nf = probe_media(media_dir / "does_not_exist.mp4")
    assert nf.status is ProbeStatus.NOT_FOUND
    with pytest.raises(RuntimeError):
        availability_from_probe(nf)
    tool = probe_media(media_dir / "video_audio.mp4", ffprobe_bin="ffprobe-definitely-not-installed")
    assert tool.status is ProbeStatus.TOOL_MISSING
    with pytest.raises(RuntimeError):
        availability_from_probe(tool)


def test_timeout_maps_to_corrupted_not_missing() -> None:
    assert set(availability_from_probe(MediaProbe(status=ProbeStatus.TIMEOUT)).values()) == {
        ModalityStatus.CORRUPTED
    }
