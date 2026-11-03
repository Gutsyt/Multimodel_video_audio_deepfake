"""Visual detector: frozen CLIP ViT-L/14 image features + trained, calibrated linear head.

Recipe: Ojha et al., "Towards Universal Fake Image Detectors that Generalize Across Generative
Models" (CVPR 2023): a linear probe on frozen CLIP features generalises far better than a
fine-tuned classifier. Head trained on FaceForensics++ (c23), Celeb-DF/FakeAVCeleb frames and
WildDeepfake; see docs/MODEL_REPORT.md.
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

BACKBONE = "openai/clip-vit-large-patch14"
HEAD_FILE = "visual_head.npz"


class ClipFaceDetector:
    def __init__(self, device: str = "cpu", batch_size: int = 16, with_head: bool = True) -> None:
        self._device, self._batch, self._with_head = device, batch_size, with_head
        self._model: Any = None
        self._processor: Any = None
        self.head: LinearHead | None = None
        self.layer = -1  # hidden-state index used as the feature (CLS token); set by the head
        self.output_semantics: dict[str, Any] = {}
        self.model_card = _card({})

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import CLIPImageProcessor, CLIPVisionModel
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
            self._processor = CLIPImageProcessor.from_pretrained(BACKBONE)
            self._model = CLIPVisionModel.from_pretrained(BACKBONE).to(self._device).eval()  # type: ignore[arg-type,unused-ignore]
        except Exception as exc:
            raise ModelLoadError("download_failed", f"could not load {BACKBONE}: {type(exc).__name__}") from exc
        self.output_semantics = {"score_class": "fake", "calibrated": True,
                                 "higher_means": "calibrated probability that the face is manipulated"}  # fmt: skip

    def embed(self, faces_rgb: Sequence[NDArray[np.uint8]], layers: Sequence[int] | None = None) -> dict[int, NDArray[np.float32]]:
        """CLS-token features per hidden-state layer (default: the head's layer)."""
        import torch

        if self._model is None:
            raise ModelLoadError("not_loaded", "call load() before scoring")
        layers = list(layers or [self.layer])
        out: dict[int, list[NDArray[np.float32]]] = {k: [] for k in layers}
        with torch.inference_mode():
            for i in range(0, len(faces_rgb), self._batch):
                px = self._processor(images=list(faces_rgb[i : i + self._batch]), return_tensors="pt")
                hs = self._model(pixel_values=px["pixel_values"].to(self._device), output_hidden_states=True).hidden_states
                for k in layers:
                    out[k].append(hs[k][:, 0].float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def score_faces(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[float]:
        return [it.score for it in self.score_faces_detailed(faces_rgb)]

    def score_faces_detailed(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[ScoredItem]:
        if self.head is None:
            raise ModelLoadError("not_loaded", "no trained head loaded")
        probs = self.head.prob(self.embed(faces_rgb)[self.layer])
        return [ScoredItem(score=float(p)) for p in probs]


def _card(meta: dict[str, Any]) -> ModelCard:
    return ModelCard(
        name="CLIP-L/14 + linear probe",
        version=f"{BACKBONE} | head {meta.get('trained_utc', 'untrained')}",
        modality=Modality.VISUAL,
        license="Backbone: MIT (OpenAI CLIP). Head: trained in this project.",
        source_url=f"https://huggingface.co/{BACKBONE}",
        citation="Radford et al., CLIP (ICML 2021); Ojha et al., UniversalFakeDetect (CVPR 2023)",
        trained_on=meta.get("trained_on", "-"),
        limitations=tuple(meta.get("limitations", ())),
        metrics=meta.get("metrics"),
    )
