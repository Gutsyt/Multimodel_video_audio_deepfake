"""Server settings, read from AVDD_* environment variables (safe local defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_CONTAINERS = frozenset({"mov", "mp4", "matroska", "webm", "avi"})  # video (also audio-only m4a/webm/mkv)
ALLOWED_AUDIO_CONTAINERS = frozenset({"wav", "mp3", "flac", "ogg", "aac"})
ALLOWED_IMAGE_CONTAINERS = frozenset({"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe"})
ALLOWED_IMAGE_CODECS = frozenset({"mjpeg", "png", "webp", "bmp"})

# File extensions shown to users and used for client-side pre-checks only. The server never trusts
# an extension or MIME type: it decides by parsing the content with ffprobe.
ACCEPTED_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "video": (".mp4", ".mov", ".mkv", ".webm", ".avi"),
    "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"),
    "image": (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


@dataclass(frozen=True)
class Settings:
    """Limits and locations. Raw uploads live only in ``upload_dir`` while a job runs."""

    data_dir: Path = Path("app_data")
    max_upload_bytes: int = 100 * 1024 * 1024
    max_duration_s: float = 60.0
    max_pending: int = 4
    n_frames: int = 12
    history_max_items: int = 200
    audio_model: str = "xlsr"  # xlsr | none
    video_model: str = "clip"  # clip | none
    device: str = "cpu"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    frontend_dist: Path | None = None
    enable_docs: bool = True
    allowed_containers: frozenset[str] = field(default=ALLOWED_CONTAINERS)
    allowed_audio_containers: frozenset[str] = field(default=ALLOWED_AUDIO_CONTAINERS)
    allowed_image_containers: frozenset[str] = field(default=ALLOWED_IMAGE_CONTAINERS)
    max_image_pixels: int = 40_000_000

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads_tmp"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "analyses.sqlite3"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        origins = env("AVDD_CORS_ORIGINS")
        dist = env("AVDD_FRONTEND_DIST")
        return cls(
            data_dir=Path(env("AVDD_DATA_DIR", "app_data")),
            max_upload_bytes=_env_int("AVDD_MAX_UPLOAD_MB", 100) * 1024 * 1024,
            max_duration_s=float(_env_int("AVDD_MAX_DURATION_S", 60)),
            max_pending=_env_int("AVDD_MAX_PENDING", 4),
            n_frames=_env_int("AVDD_FRAMES", 12),
            history_max_items=_env_int("AVDD_HISTORY_MAX", 200),
            max_image_pixels=_env_int("AVDD_MAX_IMAGE_MEGAPIXELS", 40) * 1_000_000,
            audio_model=env("AVDD_AUDIO_MODEL", "xlsr") or "xlsr",
            video_model=env("AVDD_VIDEO_MODEL", "clip") or "clip",
            device=env("AVDD_DEVICE", "cpu") or "cpu",
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip())
            if origins
            else cls.cors_origins,
            frontend_dist=Path(dist) if dist else None,
            enable_docs=env("AVDD_ENABLE_DOCS", "1") != "0",
        )
