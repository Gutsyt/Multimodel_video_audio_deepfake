"""Trained linear head over frozen backbone embeddings, with Platt calibration and thresholds.

Fitted by ``scripts/train_heads.py``; stored as a small ``.npz`` in ``avdd/assets``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ASSETS = Path(__file__).resolve().parent.parent / "assets"


class LinearHead:
    def __init__(self, path: str | Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.mean, self.scale = z["mean"], z["scale"]
        self.w, self.b = z["w"], float(z["b"])
        self.cal_a, self.cal_b = float(z["cal_a"]), float(z["cal_b"])
        self.meta: dict[str, Any] = json.loads(str(z["meta"]))

    @property
    def t_fake(self) -> float:
        return float(self.meta["t_fake"])

    @property
    def t_real(self) -> float:
        return float(self.meta["t_real"])

    def prob(self, emb: NDArray[np.float32]) -> NDArray[np.float64]:
        """Calibrated P(fake) per row of ``emb`` (rows are L2-normalised first)."""
        x = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)
        logit = ((x - self.mean) / self.scale) @ self.w + self.b
        out: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-(self.cal_a * logit + self.cal_b)))
        return out
