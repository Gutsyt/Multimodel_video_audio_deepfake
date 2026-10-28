"""Audio detector: frozen wav2vec 2.0 XLS-R (300M) features + trained, calibrated linear head.

Recipe: Pascu et al., "Towards generalisable and calibrated audio deepfake detection with
self-supervised representations" (Interspeech 2024): a logistic regression on frozen SSL
features generalises to in-the-wild audio far better than AASIST-style end-to-end models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from avdd.common.errors import ModelLoadError
from avdd.common.head import ASSETS, LinearHead
from avdd.common.model_card import ModelCard
from avdd.common.protocols import ScoredItem
from avdd.common.types import Modality

BACKBONE = "facebook/wav2vec2-xls-r-300m"
HEAD_FILE = "audio_head.npz"
MAX_LAYERS = 12  # only the first transformer layers are run; spoofing cues live in the lower half


class SslAudioDetector:
    def __init__(self, device: str = "cpu", batch_size: int = 4, with_head: bool = True) -> None:
        self._device, self._batch, self._with_head = device, batch_size, with_head
        self._model: Any = None
        self._fe: Any = None
        self.head: LinearHead | None = None
        self.layer = 8
        self.output_semantics: dict[str, Any] = {}
        self.model_card = _card({})

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        except ImportError as exc:
            raise ModelLoadError("deps_missing", "torch/transformers are not installed") from exc
        if self._with_head:
            try:
                self.head = LinearHead(ASSETS / HEAD_FILE)
            except OSError as exc:
                raise ModelLoadError("head_missing", f"{HEAD_FILE} not found; run scripts/train_heads.py") from exc
            self.layer = int(self.head.meta["layer"])
            self.model_card = _card(self.head.meta)
        try:
            self._fe = Wav2Vec2FeatureExtractor.from_pretrained(BACKBONE)
            model = Wav2Vec2Model.from_pretrained(BACKBONE)
        except Exception as exc:
            raise ModelLoadError("download_failed", f"could not load {BACKBONE}: {type(exc).__name__}") from exc
        model.encoder.layers = model.encoder.layers[:MAX_LAYERS]
        self._model = model.to(self._device).eval()  # type: ignore[arg-type]
        self.output_semantics = {"score_class": "spoof", "calibrated": True,
                                 "higher_means": "calibrated probability that the speech is synthetic"}  # fmt: skip

    def embed(self, windows: Sequence[NDArray[np.float32]], layers: Sequence[int] | None = None) -> dict[int, NDArray[np.float32]]:
        """Time-averaged hidden states per layer (default: the head's layer)."""
        import torch

        if self._model is None:
            raise ModelLoadError("not_loaded", "call load() before scoring")
        layers = list(layers or [self.layer])
        out: dict[int, list[NDArray[np.float32]]] = {k: [] for k in layers}
        with torch.inference_mode():
            for i in range(0, len(windows), self._batch):
                x = self._fe(list(windows[i : i + self._batch]), sampling_rate=16_000, return_tensors="pt")
                hs = self._model(x["input_values"].to(self._device), output_hidden_states=True).hidden_states
                for k in layers:
                    out[k].append(hs[k].mean(dim=1).float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def score_windows(self, windows: Sequence[NDArray[np.float32]]) -> list[float]:
        return [it.score for it in self.score_windows_detailed(windows)]

    def score_windows_detailed(self, windows: Sequence[NDArray[np.float32]]) -> list[ScoredItem]:
        if self.head is None:
            raise ModelLoadError("not_loaded", "no trained head loaded")
        probs = self.head.prob(self.embed(windows)[self.layer])
        return [ScoredItem(score=float(p)) for p in probs]


def _card(meta: dict[str, Any]) -> ModelCard:
    return ModelCard(
        name="XLS-R 300M + linear probe",
        version=f"{BACKBONE} | head {meta.get('trained_utc', 'untrained')}",
        modality=Modality.AUDIO,
        license="Backbone: Apache-2.0 (Meta wav2vec 2.0 XLS-R). Head: trained in this project.",
        source_url=f"https://huggingface.co/{BACKBONE}",
        citation="Babu et al., XLS-R (2021); Pascu et al., Interspeech 2024",
        trained_on=meta.get("trained_on", "-"),
        limitations=tuple(meta.get("limitations", ())),
        metrics=meta.get("metrics"),
    )
