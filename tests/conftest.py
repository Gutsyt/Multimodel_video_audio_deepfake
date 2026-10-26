"""Synthetic media fixtures.

These files are ffmpeg test patterns and sine tones. They exist ONLY to test
software behaviour (parsing, missing streams, corruption handling). They contain
no faces or speech and say nothing about deepfake detection performance.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _run(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, timeout=60)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with: video+audio, video-only, audio-only, garbage, truncated, text files."""
    d = tmp_path_factory.mktemp("synthetic_media")
    v = ["-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=25"]
    a = ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=16000"]
    _run([*v, *a, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
          str(d / "video_audio.mp4")])  # fmt: skip
    _run([*v, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "video_only.mp4")])
    _run([*a, str(d / "audio_only.wav")])
    (d / "garbage.mp4").write_bytes(bytes(range(256)) * 64)
    full = (d / "video_audio.mp4").read_bytes()
    (d / "truncated.mp4").write_bytes(full[: len(full) * 2 // 5])
    (d / "notes.txt").write_text("not media\n")
    return d


@pytest.fixture(scope="session")
def face_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Clips built from a public-domain NASA photo (scikit-image's `astronaut`) + sine tones.

    SOFTWARE-TEST FIXTURES ONLY: a still face looped as video is not a deepfake or a real recording.
    """
    cv2 = pytest.importorskip("cv2")
    skimage_data = pytest.importorskip("skimage.data")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("face_media")
    cv2.imwrite(str(d / "face.png"), skimage_data.astronaut()[:, :, ::-1])
    img = ["-loop", "1", "-framerate", "10", "-i", str(d / "face.png")]
    enc = ["-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    sine = ["-f", "lavfi", "-i", "sine=frequency=300:duration=3:sample_rate=16000"]
    silence = ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono"]
    _run([*img, *sine, *enc, "-c:a", "aac", "-shortest", str(d / "face_audio.mp4")])
    _run([*img, *enc, str(d / "face_noaudio.mp4")])
    _run([*img, *silence, *enc, "-c:a", "aac", "-shortest", str(d / "face_silent.mp4")])
    return d


@pytest.fixture(scope="session")
def kinds_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Audio-only, image-only and 'video without a face' files for the optional-modality tests.

    SOFTWARE-TEST FIXTURES ONLY (sine tones, a public-domain photo, ffmpeg test patterns).
    """
    cv2 = pytest.importorskip("cv2")
    skimage_data = pytest.importorskip("skimage.data")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("kinds_media")
    face = skimage_data.astronaut()[:, :, ::-1]
    cv2.imwrite(str(d / "face.jpg"), face)
    cv2.imwrite(str(d / "face.png"), face)
    cv2.imwrite(str(d / "face.webp"), face)
    cv2.imwrite(str(d / "face.bmp"), face)
    import numpy as np

    noise = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(d / "noface.jpg"), noise)
    sine = ["-f", "lavfi", "-i", "sine=frequency=300:duration=3:sample_rate=16000"]
    _run([*sine, str(d / "tone.mp3")])
    _run([*sine, str(d / "tone.flac")])
    _run([*sine, "-c:a", "libvorbis", str(d / "tone.ogg")])
    _run([*sine, "-c:a", "aac", str(d / "tone.m4a")])
    pattern = ["-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10"]
    _run([*pattern, *sine, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
          str(d / "noface_audio.mp4")])  # fmt: skip
    cv2.imwrite(str(d / "face.tiff"), face)
    _run([*pattern, str(d / "animated.gif")])
    (d / "empty.wav").write_bytes(b"")
    (d / "fake.jpg").write_text("this is not an image\n")
    (d / "truncated.jpg").write_bytes((d / "face.jpg").read_bytes()[:300])
    (d / "truncated.mp3").write_bytes((d / "tone.mp3").read_bytes()[:60])
    return d
