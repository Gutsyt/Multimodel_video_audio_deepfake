"""REAL MODEL INFERENCE smoke tests (no stubs).

These load the vendored pretrained AASIST weights and the Hugging Face visual model and check
PLUMBING and plausibility only: models load, outputs are finite and in range, the class mapping
reported in results matches the checkpoint's own config, the pipeline runs end to end.

They are NOT accuracy tests. They assert nothing about whether a score is "right" and cannot:
there is no labelled evaluation data here. Skipped automatically when torch/transformers or the
Hugging Face download are unavailable; deselect with ``-m "not real_models"``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("skimage")

from avdd.audio.aasist_detector import CLASS_LABELS, SPOOF_INDEX, AasistDetector
from avdd.common.errors import ModelLoadError
from avdd.pipeline.analyze import analyze_media
from avdd.video.faces import HaarFaceDetector
from avdd.video.hf_detector import DEFAULT_MODEL_ID, DEFAULT_REVISION, HfImageDetector
from tests.conftest import requires_ffmpeg

pytestmark = [pytest.mark.real_models, requires_ffmpeg]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def aasist() -> AasistDetector:
    det = AasistDetector("AASIST")
    try:
        det.load()
    except ModelLoadError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"AASIST could not be loaded: {exc.code}")
    return det


@pytest.fixture(scope="module")
def hf() -> HfImageDetector:
    det = HfImageDetector(DEFAULT_MODEL_ID, DEFAULT_REVISION)
    try:
        det.load()
    except ModelLoadError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"visual model could not be loaded: {exc.code}")
    return det


def test_aasist_reports_both_logits_and_the_score_is_the_spoof_probability(aasist: AasistDetector) -> None:
    rng = np.random.default_rng(0)
    win = (rng.standard_normal(64_600) * 0.05).astype(np.float32)
    (item,) = aasist.score_windows_detailed([win])
    assert item.logits is not None and item.probs is not None
    assert list(item.probs) == list(CLASS_LABELS) and CLASS_LABELS[SPOOF_INDEX] == "spoof"
    assert sum(item.probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert item.score == pytest.approx(item.probs["spoof"]) and 0.0 <= item.score <= 1.0
    assert item.probs["spoof"] == pytest.approx(
        math.exp(item.logits["spoof"]) / (math.exp(item.logits["spoof"]) + math.exp(item.logits["bona_fide"])), abs=1e-5)
    assert aasist.score_windows([win]) == [pytest.approx(item.score)]  # deterministic, same path


def test_visual_model_label_mapping_comes_from_the_checkpoint_config(hf: HfImageDetector) -> None:
    sem = hf.output_semantics
    assert sem["class_labels"] == ["Realism", "Deepfake"]  # the checkpoint's own id2label
    assert sem["score_class"] == "Deepfake" and sem["score_class_index"] == 1


def test_visual_model_scores_a_face_crop_with_both_class_probabilities(hf: HfImageDetector) -> None:
    import cv2

    img = cv2.imread(str(ROOT / "face.jpg"))
    assert img is not None
    face = HaarFaceDetector().detect(img)[0]
    from avdd.video.faces import crop_face_rgb

    (item,) = hf.score_faces_detailed([crop_face_rgb(img, face)])
    assert item.probs is not None and set(item.probs) == {"Realism", "Deepfake"}
    assert sum(item.probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert item.score == pytest.approx(item.probs["Deepfake"]) and math.isfinite(item.score)


def test_pipeline_with_real_models_on_an_image_and_on_audio(
    aasist: AasistDetector, hf: HfImageDetector, kinds_dir: Path
) -> None:
    image = analyze_media(ROOT / "face.jpg", aasist, hf, HaarFaceDetector())
    assert image["is_test_fixture"] is False and image["assessment"]["basis"] == "visual_only"
    assert image["modalities"]["audio"]["status"] == "missing" and image["fusion"]["fused_score"] is None
    v = image["modalities"]["visual"]
    assert v["status"] == "available" and v["diagnostics"]["output_semantics"]["score_class"] == "Deepfake"

    tone = analyze_media(kinds_dir / "tone.flac", aasist, hf, HaarFaceDetector())
    a = tone["modalities"]["audio"]
    assert a["status"] in {"available", "unreliable"}
    if a["status"] == "available":
        assert a["diagnostics"]["output_semantics"]["score_class"] == "spoof"
        assert tone["assessment"]["basis"] == "audio_only"
    assert tone["modalities"]["visual"]["status"] == "missing"
