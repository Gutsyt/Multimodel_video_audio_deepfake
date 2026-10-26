"""Wrapper for a pretrained Hugging Face image-classification deepfake detector (INFERENCE ONLY)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from avdd.common.errors import ModelLoadError
from avdd.common.model_card import ModelCard
from avdd.common.protocols import ScoredItem
from avdd.common.types import Modality

DEFAULT_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
# Revision copied from a Hugging Face page snapshot seen during research; NOT yet confirmed by a
# download. If it fails, run scripts/probe_video_model.py --revision main and pin the printed hash.
DEFAULT_REVISION = "28430e0e0573cdec1d20fc0561747db026b3569c"


class HfImageDetector:
    """Scores face crops with a ViT image classifier; the 'fake' class is found from config labels."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, revision: str | None = DEFAULT_REVISION,
                 device: str = "cpu", batch_size: int = 8) -> None:  # fmt: skip
        self.model_id, self.revision, self._device, self._batch = model_id, revision, device, batch_size
        self._model: Any = None
        self._processor: Any = None
        self._fake_idx = -1
        self._id2label: dict[int, str] = {}
        self.output_semantics: dict[str, Any] = {}
        self.model_card = ModelCard(
            name=model_id.split("/")[-1],
            version=f"{model_id}@{revision or 'main'}",
            modality=Modality.VISUAL,
            license="Apache-2.0 (per the model card; training-data license undisclosed)",
            source_url=f"https://huggingface.co/{model_id}",
            citation=(
                "prithivMLmods, Deep-Fake-Detector-v2-Model (Hugging Face, 2025); fine-tuned from "
                "google/vit-base-patch16-224-in21k (Dosovitskiy et al., An Image is Worth 16x16 "
                "Words, ICLR 2021)"
            ),
            trained_on="Undisclosed: the model card says only 'a curated dataset of real and deepfake images'",
            limitations=(
                "Pretrained inference only; no training or fine-tuning in this project.",
                "Image-level classifier applied to individual face crops, not a video/temporal model.",
                (
                    "Training data, its provenance and license are not disclosed, so what manipulations "
                    "it covers (GAN faces vs. face swaps) is unknown."
                ),
                (
                    "The model card itself warns of poor generalization to other datasets/domains and "
                    "degradation on low-resolution or compressed images."
                ),
                "Vendor-reported metrics (e.g. 'precision-92.12' tag) are not reproduced here.",
                "Scores are raw softmax outputs: uncalibrated and not validated on this project's data.",
            ),
        )

    def load(self) -> dict[str, Any]:
        """Download/load the model; returns {'id2label', 'fake_index', 'commit'} for logging."""
        try:
            import torch  # type: ignore[import-not-found,unused-ignore]  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ModelLoadError("deps_missing", "torch/transformers are not installed") from exc
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id, revision=self.revision)
            model = AutoModelForImageClassification.from_pretrained(self.model_id, revision=self.revision)
        except Exception as exc:
            raise ModelLoadError("download_failed", f"could not load {self.model_id}: {type(exc).__name__}") from exc
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        fakes = [i for i, lab in id2label.items() if "fake" in lab.lower()]
        if len(fakes) != 1:
            raise ModelLoadError("ambiguous_labels", f"cannot identify the 'fake' class from labels {id2label}")
        self._fake_idx = fakes[0]
        self._id2label = id2label
        self.output_semantics = {
            "class_labels": [id2label[i] for i in sorted(id2label)],
            "score_class": id2label[self._fake_idx],
            "score_class_index": self._fake_idx,
            "activation": "softmax over the classifier logits",
            "reference": (
                f"model config id2label {id2label}; the single label containing 'fake' is used "
                "(load fails if that is ambiguous)"
            ),
            "higher_means": "more weight on the fake class (not a calibrated probability)",
        }
        self._model = model.to(self._device).eval()
        return {"id2label": id2label, "fake_index": self._fake_idx,
                "commit": getattr(model.config, "_commit_hash", None)}  # fmt: skip

    @property
    def device(self) -> str:
        return self._device

    def score_faces(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[float]:
        return [item.score for item in self.score_faces_detailed(faces_rgb)]

    def score_faces_detailed(self, faces_rgb: Sequence[NDArray[np.uint8]]) -> list[ScoredItem]:
        """Per-crop score plus every class's logit and probability (for diagnostics)."""
        if self._model is None:
            raise ModelLoadError("not_loaded", "call load() before scoring")
        import torch  # type: ignore[import-not-found,unused-ignore]
        from PIL import Image

        order = sorted(self._id2label)
        out: list[ScoredItem] = []
        with torch.inference_mode():
            for i in range(0, len(faces_rgb), self._batch):
                imgs = [Image.fromarray(f) for f in faces_rgb[i : i + self._batch]]
                inputs = self._processor(images=imgs, return_tensors="pt").to(self._device)
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                for lg, pr in zip(logits.cpu().tolist(), probs.cpu().tolist(), strict=True):
                    out.append(ScoredItem(
                        score=float(pr[self._fake_idx]),
                        logits={self._id2label[k]: float(lg[k]) for k in order},
                        probs={self._id2label[k]: float(pr[k]) for k in order},
                    ))
        return out
