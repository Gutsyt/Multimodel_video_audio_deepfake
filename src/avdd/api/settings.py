"""Server settings, read from AVDD_* environment variables (safe local defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_CONTAINERS = frozenset({"mov", "mp4", "matroska", "webm", "avi"})


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
    n_frames: int = 16
    history_max_items: int = 200
    audio_model: str = "AASIST"  # AASIST | AASIST-L | none
    video_model: str = "hf"  # hf | none
    hf_model_id: str | None = None
    hf_revision: str | None = None
    device: str = "cpu"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    frontend_dist: Path | None = None
    enable_docs: bool = True
    allowed_containers: frozenset[str] = field(default=ALLOWED_CONTAINERS)

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
            n_frames=_env_int("AVDD_FRAMES", 16),
            history_max_items=_env_int("AVDD_HISTORY_MAX", 200),
            audio_model=env("AVDD_AUDIO_MODEL", "AASIST") or "AASIST",
            video_model=env("AVDD_VIDEO_MODEL", "hf") or "hf",
            hf_model_id=env("AVDD_HF_MODEL") or None,
            hf_revision=env("AVDD_HF_REVISION") or None,
            device=env("AVDD_DEVICE", "cpu") or "cpu",
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip())
            if origins
            else cls.cors_origins,
            frontend_dist=Path(dist) if dist else None,
            enable_docs=env("AVDD_ENABLE_DOCS", "1") != "0",
        )
