"""Exception types shared across the project (messages never contain file paths)."""

from __future__ import annotations


class AvddError(Exception):
    """Base class: carries a stable machine-readable ``code`` plus a safe message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AudioError(AvddError):
    """Audio could not be decoded into usable samples."""


class VideoError(AvddError):
    """Video frames could not be decoded."""


class ModelLoadError(AvddError):
    """A pretrained model could not be loaded."""
