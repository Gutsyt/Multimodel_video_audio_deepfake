"""Uniform frame sampling with OpenCV (timestamps are nominal: index / fps)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avdd.common.errors import VideoError


@dataclass(frozen=True)
class SampledFrame:
    index: int
    t_s: float | None  # None for a still image: there is no timeline
    image_bgr: NDArray[np.uint8]


MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard; the upload gate enforces the same cap


def load_image_frame(path: str | Path, max_pixels: int = MAX_IMAGE_PIXELS) -> SampledFrame:
    """Decode a still image (JPEG/PNG/WebP/BMP) into one BGR frame.

    EXIF orientation is applied by OpenCV's decoder. Raises ``VideoError``
    (image_decode_failed, image_too_large).
    """
    import cv2

    try:
        raw = np.fromfile(str(Path(path).resolve()), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    except (OSError, cv2.error) as exc:
        raise VideoError("image_decode_failed", "image could not be decoded") from exc
    if img is None:
        raise VideoError("image_decode_failed", "image could not be decoded")
    h, w = img.shape[:2]
    if h * w > max_pixels:
        raise VideoError("image_too_large", f"image exceeds {max_pixels / 1e6:g} megapixels")
    return SampledFrame(0, None, np.asarray(img, dtype=np.uint8))


def sample_frames(
    path: str | Path, n_frames: int = 16, max_seconds: float = 60.0
) -> list[SampledFrame]:
    """Return up to ``n_frames`` frames spread uniformly over the first ``max_seconds``.

    Raises ``VideoError`` (open_failed, bad_metadata, no_decodable_frames).
    """
    import cv2  # local import keeps the base install lightweight

    cap = cv2.VideoCapture(str(Path(path).resolve()))
    try:
        if not cap.isOpened():
            raise VideoError("open_failed", "video could not be opened")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise VideoError("bad_metadata", "video has no usable frame rate or frame count")
        limit = max(1, min(total, int(max_seconds * fps)))
        indices = sorted({int(i) for i in np.linspace(0, limit - 1, num=n_frames)})
        frames: list[SampledFrame] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if ok and img is not None:
                frames.append(SampledFrame(idx, idx / fps, np.asarray(img, dtype=np.uint8)))
    finally:
        cap.release()
    if not frames:
        raise VideoError("no_decodable_frames", "no frames could be decoded")
    return frames
