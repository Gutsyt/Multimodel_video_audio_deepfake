"""Face detection (YuNet, MIT, via OpenCV's FaceDetectorYN) and cropping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from avdd.common.head import ASSETS


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return self.w * self.h


class FaceBoxDetector(Protocol):
    name: str

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[FaceBox]: ...


class YuNetFaceDetector:
    name = "yunet-2023mar"

    def __init__(self, score_threshold: float = 0.8, min_size: int = 40, max_side: int = 960) -> None:
        import cv2

        self._det = cv2.FaceDetectorYN.create(
            str(ASSETS / "face_detection_yunet_2023mar.onnx"), "", (320, 320), score_threshold)
        self._min, self._max_side = min_size, max_side

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[FaceBox]:
        """Faces sorted by area (largest first), in original-image pixels."""
        import cv2

        h, w = image_bgr.shape[:2]
        s = min(1.0, self._max_side / max(h, w))
        img = cv2.resize(image_bgr, (int(w * s), int(h * s))) if s < 1.0 else image_bgr
        self._det.setInputSize((img.shape[1], img.shape[0]))
        _, raw = self._det.detect(img)
        boxes = [FaceBox(int(r[0] / s), int(r[1] / s), int(r[2] / s), int(r[3] / s), float(r[-1]))
                 for r in ([] if raw is None else raw)]  # fmt: skip
        return sorted((b for b in boxes if min(b.w, b.h) >= self._min), key=lambda b: b.area, reverse=True)


def crop_face_rgb(
    image_bgr: NDArray[np.uint8], box: FaceBox, margin: float = 0.3
) -> NDArray[np.uint8]:
    """Square crop around ``box`` expanded by ``margin`` per side, clipped to the image, as RGB."""
    h, w = image_bgr.shape[:2]
    side = int(max(box.w, box.h) * (1 + 2 * margin))
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    return np.ascontiguousarray(image_bgr[y0:y1, x0:x1, ::-1])
