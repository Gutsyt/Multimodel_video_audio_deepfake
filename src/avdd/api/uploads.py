"""Safe upload handling.

The body is streamed to a randomly named file (the client's file name is never used or stored),
with a hard byte limit enforced while streaming. Acceptance is decided by ACTUALLY PARSING the
media with ffprobe (container allowlist, streams, duration), not by extension or MIME type.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from avdd.api.errors import ApiError
from avdd.api.settings import Settings
from avdd.media.probe import MediaProbe, ProbeStatus, probe_media

_CHUNK_LIMIT_MSG = "The file is larger than the allowed upload size."


async def receive_upload(request: Request, upload_dir: Path, max_bytes: int) -> Path:
    """Stream the request body to ``upload_dir``; returns the path. Cleans up on any failure."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ApiError(413, "file_too_large", _CHUNK_LIMIT_MSG, {"max_upload_mb": round(max_bytes / 1e6, 1)})
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}.upload"
    total = 0
    try:
        with path.open("wb") as fh:
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_bytes:
                    raise ApiError(413, "file_too_large", _CHUNK_LIMIT_MSG,
                                   {"max_upload_mb": round(max_bytes / 1e6, 1)})
                await run_in_threadpool(fh.write, chunk)
        if total == 0:
            raise ApiError(400, "empty_upload", "The upload was empty.")
    except ClientDisconnect as exc:
        path.unlink(missing_ok=True)
        raise ApiError(400, "upload_interrupted", "The upload was interrupted.") from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def validate_media(path: Path, settings: Settings) -> MediaProbe:
    """Parse the file with ffprobe and enforce container, stream and duration rules."""
    probe = probe_media(path, timeout_s=30.0)
    if probe.status is ProbeStatus.TOOL_MISSING:
        raise ApiError(503, "tool_missing", "ffprobe is not installed on the server.")
    if probe.status is not ProbeStatus.OK:
        raise ApiError(422, "unreadable_media", "The file could not be parsed as audio or video; it may be corrupted.")
    detected = {c.strip() for c in (probe.container or "").split(",")}
    if not detected & settings.allowed_containers:
        raise ApiError(415, "unsupported_format",
                       "Unsupported file type. Upload a video in MP4, MOV, MKV, WebM or AVI format.",
                       {"detected_container": probe.container, "accepted": sorted(settings.allowed_containers)})
    if probe.duration_s is None or probe.duration_s <= 0:
        raise ApiError(422, "unknown_duration", "The media duration could not be determined.")
    if probe.duration_s > settings.max_duration_s:
        raise ApiError(422, "duration_exceeded",
                       f"The media is longer than the {settings.max_duration_s:g} s limit.",
                       {"duration_s": round(probe.duration_s, 1), "max_duration_s": settings.max_duration_s})
    return probe
