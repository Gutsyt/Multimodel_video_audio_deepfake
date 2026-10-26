"""Inference wrapper for the pretrained AASIST / AASIST-L audio anti-spoofing models.

INFERENCE ONLY: the weights are the upstream checkpoints (trained by the AASIST authors
on ASVspoof 2019 LA). Nothing is trained or fine-tuned here.

Output convention (verified in upstream ``main.py``): the network returns two logits;
index 1 is the *bona fide* score, so index 0 is the spoof class. We report
``softmax(logits)[0]`` as a RAW spoof score in [0, 1]. It is not a calibrated probability.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from avdd.common.errors import ModelLoadError
from avdd.common.model_card import ModelCard
from avdd.common.types import Modality

UPSTREAM_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"

# model_config blocks copied verbatim from upstream config/AASIST.conf and config/AASIST-L.conf
_CONFIGS: dict[str, dict[str, Any]] = {
    "AASIST": {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [
            70,
            [
                1,
                32
            ],
            [
                32,
                32
            ],
            [
                32,
                64
            ],
            [
                64,
                64
            ]
        ],
        "gat_dims": [
            64,
            32
        ],
        "pool_ratios": [
            0.5,
            0.7,
            0.5,
            0.5
        ],
        "temperatures": [
            2.0,
            2.0,
            100.0,
            100.0
        ]
    },
    "AASIST-L": {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [
            70,
            [
                1,
                32
            ],
            [
                32,
                32
            ],
            [
                32,
                24
            ],
            [
                24,
                24
            ]
        ],
        "gat_dims": [
            24,
            32
        ],
        "pool_ratios": [
            0.4,
            0.5,
            0.7,
            0.5
        ],
        "temperatures": [
            2.0,
            2.0,
            100.0,
            100.0
        ]
    }
}

_WEIGHT_FILES = {"AASIST": "AASIST.pth", "AASIST-L": "AASIST-L.pth"}


def default_third_party_dir() -> Path:
    env = os.environ.get("AVDD_THIRD_PARTY_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "third_party"


class AasistDetector:
    """Loads vendored AASIST code + weights and scores 64,600-sample windows on CPU/GPU."""

    def __init__(
        self,
        variant: str = "AASIST",
        third_party_dir: Path | None = None,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        if variant not in _CONFIGS:
            raise ModelLoadError("unknown_variant", f"variant must be one of {sorted(_CONFIGS)}")
        self.variant = variant
        self._dir = (third_party_dir or default_third_party_dir()) / "aasist"
        self._device = device
        self._batch = batch_size
        self._model: Any = None
        self.model_card = ModelCard(
            name=variant,
            version=f"clovaai/aasist@{UPSTREAM_COMMIT[:7]} ({_WEIGHT_FILES[variant]})",
            modality=Modality.AUDIO,
            license="MIT (NAVER Corp.), code and weights as distributed in the upstream repository",
            source_url="https://github.com/clovaai/aasist",
            citation=(
                "Jung et al., AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal "
                "Graph Attention Networks, ICASSP 2022, pp. 6367-6371"
            ),
            trained_on="ASVspoof 2019 Logical Access (train partition), per upstream README",
            limitations=(
                "Pretrained inference only; no training or fine-tuning in this project.",
                (
                    "Trained on clean English TTS/voice-conversion attacks (ASVspoof 2019 LA, derived "
                    "from VCTK). A third-party benchmark reports near-chance equal error rates "
                    "(about 43-44%) on in-the-wild audio versus <1% in-domain; not reproduced here."
                ),
                "Meaningful only for speech: in our smoke test, pure noise scored as spoof.",
                "Scores are raw softmax outputs: uncalibrated and not validated on this project's data.",
            ),
        )

    def load(self) -> None:
        """Import the vendored model code and load weights (weights_only=True)."""
        try:
            import torch  # type: ignore[import-not-found,unused-ignore]
        except ImportError as exc:
            raise ModelLoadError("torch_missing", "PyTorch is not installed") from exc
        code, weights = self._dir / "AASIST.py", self._dir / "weights" / _WEIGHT_FILES[self.variant]
        if not code.is_file() or not weights.is_file():
            raise ModelLoadError("files_missing", "vendored AASIST files not found (third_party/aasist)")
        spec = importlib.util.spec_from_file_location("avdd_vendored_aasist", code)
        if spec is None or spec.loader is None:
            raise ModelLoadError("import_failed", "cannot import vendored AASIST code")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            model = module.Model(_CONFIGS[self.variant])
            state = torch.load(weights, map_location=self._device, weights_only=True)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ModelLoadError("weights_incompatible", "AASIST weights failed to load") from exc
        self._model = model.to(self._device).eval()

    def score_windows(self, windows: Sequence[NDArray[np.float32]]) -> list[float]:
        """Return raw spoof scores (softmax mass on class 0) for each 64,600-sample window."""
        if self._model is None:
            raise ModelLoadError("not_loaded", "call load() before scoring")
        import torch  # type: ignore[import-not-found,unused-ignore]

        scores: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(windows), self._batch):
                batch = torch.from_numpy(np.stack(windows[i : i + self._batch])).to(self._device)
                _, logits = self._model(batch)
                scores.extend(torch.softmax(logits, dim=1)[:, 0].cpu().tolist())
        return [float(s) for s in scores]
