"""Safe ffprobe wrapper: parses real media metadata instead of trusting extensions.

No shell is used, every call has a timeout, and error text is scrubbed of the
input path so it can be logged or returned without leaking filesystem layout.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 300


class ProbeStatus(str, Enum):
    """Outcome of probing a file."""

    OK = "ok"
    FAILED = "failed"  # ffprobe ran but could not parse the file
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    TOOL_MISSING = "tool_missing"  # ffprobe is not installed


@dataclass(frozen=True)
class MediaProbe:
    """Parsed container/stream facts for one media file."""

    status: ProbeStatus
    container: str | None = None
    duration_s: float | None = None
    has_video: bool = False
    has_audio: bool = False
    n_video_streams: int = 0
    n_audio_streams: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_fraction(value: Any) -> float | None:
    """Parse ffprobe rates such as ``30000/1001``; return None for 0/0 or junk."""
    if not isinstance(value, str) or "/" not in value:
        return _to_float(value)
    num_s, _, den_s = value.partition("/")
    num, den = _to_float(num_s), _to_float(den_s)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _scrub(text: str, path: str) -> str:
    return text.replace(path, "<input>").strip()[:_MAX_ERROR_CHARS]


def ffprobe_version(ffprobe_bin: str = "ffprobe") -> str | None:
    """Return the first line of ``ffprobe -version`` or None if unavailable."""
    if shutil.which(ffprobe_bin) is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe_bin, "-version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.stdout.splitlines()
    return lines[0] if lines else None


def probe_media(
    path: str | Path, timeout_s: float = 30.0, ffprobe_bin: str = "ffprobe"
) -> MediaProbe:
    """Probe ``path`` with ffprobe and return structured facts.

    Never raises for bad media: problems are reported through ``status``/``error``.
    Attached cover art (e.g. in MP3) is not counted as a video stream.
    """
    p = Path(path)
    if not p.is_file():
        return MediaProbe(status=ProbeStatus.NOT_FOUND, error="file not found")
    if shutil.which(ffprobe_bin) is None:
        return MediaProbe(status=ProbeStatus.TOOL_MISSING, error="ffprobe not found on PATH")

    abs_path = os.path.abspath(p)
    cmd = [
        ffprobe_bin, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-i", abs_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out after %.1fs", timeout_s)
        return MediaProbe(status=ProbeStatus.TIMEOUT, error=f"ffprobe timed out after {timeout_s}s")
    except OSError as exc:
        return MediaProbe(status=ProbeStatus.TOOL_MISSING, error=_scrub(str(exc), abs_path))

    if proc.returncode != 0:
        return MediaProbe(
            status=ProbeStatus.FAILED,
            error=_scrub(proc.stderr or "ffprobe failed", abs_path) or "ffprobe failed",
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return MediaProbe(status=ProbeStatus.FAILED, error="ffprobe returned unparsable output")

    streams = data.get("streams") or []
    video = [
        s for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not video and not audio:
        return MediaProbe(status=ProbeStatus.FAILED, error="no audio or video streams found")

    fmt = data.get("format") or {}
    v0 = video[0] if video else {}
    a0 = audio[0] if audio else {}
    duration = _to_float(fmt.get("duration"))
    if duration is None:
        duration = _to_float(v0.get("duration")) or _to_float(a0.get("duration"))

    return MediaProbe(
        status=ProbeStatus.OK,
        container=fmt.get("format_name"),
        duration_s=duration,
        has_video=bool(video),
        has_audio=bool(audio),
        n_video_streams=len(video),
        n_audio_streams=len(audio),
        width=_to_int(v0.get("width")),
        height=_to_int(v0.get("height")),
        fps=_parse_fraction(v0.get("avg_frame_rate")) or _parse_fraction(v0.get("r_frame_rate")),
        video_codec=v0.get("codec_name"),
        audio_codec=a0.get("codec_name"),
        sample_rate=_to_int(a0.get("sample_rate")),
        channels=_to_int(a0.get("channels")),
    )
