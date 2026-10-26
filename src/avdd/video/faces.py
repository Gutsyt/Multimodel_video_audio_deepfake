"""Face detection (OpenCV Haar cascade, bundled with OpenCV) and cropping.

Haar is a weak, frontal-face detector chosen for zero extra downloads. Upgrade path
(not implemented): YuNet (MIT) via cv2.FaceDetectorYN, which needs a Git-LFS model file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


class FaceBoxDetector(Protocol):
    name: str

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[FaceBox]: ...


class HaarFaceDetector:
    name = "opencv-haar-frontalface-default"

    def __init__(self, min_size: int = 48, scale_factor: float = 1.1, min_neighbors: int = 5,
                 max_side: int = 960) -> None:  # fmt: skip
        import cv2

        path = str(cv2.data.haarcascades) + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined,unused-ignore]
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError("Haar cascade file could not be loaded")
        self._min, self._scale, self._neighbors, self._max_side = (
            min_size, scale_factor, min_neighbors, max_side)  # fmt: skip

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[FaceBox]:
        """Return faces sorted by area (largest first); boxes are in original-image pixels."""
        import cv2

        h, w = image_bgr.shape[:2]
        s = min(1.0, self._max_side / max(h, w))
        img = cv2.resize(image_bgr, (int(w * s), int(h * s))) if s < 1.0 else image_bgr
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        raw = self._cascade.detectMultiScale(gray, self._scale, self._neighbors,
                                             minSize=(self._min, self._min))  # fmt: skip
        boxes = [FaceBox(int(x / s), int(y / s), int(bw / s), int(bh / s)) for x, y, bw, bh in raw]
        return sorted(boxes, key=lambda b: b.area, reverse=True)


def crop_face_rgb(
    image_bgr: NDArray[np.uint8], box: FaceBox, margin: float = 0.3
) -> NDArray[np.uint8]:
    """Square crop around ``box`` expanded by ``margin`` per side, clipped to the image, as RGB."""
    h, w = image_bgr.shape[:2]
    side = int(max(box.w, box.h) * (1 + 2 * margin))
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    crop = image_bgr[y0:y1, x0:x1, ::-1]  # BGR -> RGB
    return np.ascontiguousarray(crop)
