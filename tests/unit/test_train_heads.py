"""Embedding bookkeeping in scripts/train_heads.py: no source may lose its genuine or its fake samples."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import train_heads as th


class NoFaces:
    def detect(self, img: Any) -> list[Any]:
        return []


class FakeEmbedder:
    def embed(self, xs: list[Any], layers: tuple[int, ...]) -> dict[int, Any]:
        return {k: np.zeros((len(xs), 4), np.float32) for k in layers}


def _wdf(root: Path) -> Path:
    """Miniature WildDeepfake layout: {real,fake}_{train,test}/N.tar.gz, each with 2 sequences of 2 frames."""
    root.mkdir(parents=True)
    img = tmp = root / "f.png"
    cv2.imwrite(str(img), np.full((32, 32, 3), 128, np.uint8))
    for split in ("fake_test", "fake_train", "real_test", "real_train"):
        d = root / "WildDeepfake" / "deepfake_in_the_wild" / split
        d.mkdir(parents=True)
        with tarfile.open(d / "1.tar.gz", "w:gz") as t:
            for seq in ("a", "b"):
                for i in range(2):
                    t.add(tmp, arcname=f"./1/{split.split('_')[0]}/{seq}/{i}.png")
    return root


def test_every_source_keeps_both_classes(tmp_path: Path) -> None:
    data, cache = _wdf(tmp_path / "data"), tmp_path / "cache"
    cache.mkdir()
    th.embed_stream(th.visual_items(data, NoFaces(), 2), FakeEmbedder(), (16,), cache, "vis")
    for src in ("wdf", "wdf_test"):
        y = np.load(cache / f"vis_{src}.npz")["y"]
        assert set(y.tolist()) == {0, 1}, src  # the old path order saved fake-only files and dropped the reals
    assert th.finished(cache, "vis") == {"wdf", "wdf_test"}


def test_interleaved_source_is_an_error_not_silent_loss(tmp_path: Path) -> None:
    items = iter([("a", 0, 0, "g1"), ("b", 0, 0, "g2"), ("a", 0, 1, "g3")])
    with pytest.raises(RuntimeError, match="not contiguous"):
        th.embed_stream(items, FakeEmbedder(), (16,), tmp_path, "vis")


def test_single_class_cache_counts_as_unfinished(tmp_path: Path) -> None:
    np.savez(tmp_path / "vis_wdf.npz", y=np.ones(5), g=np.array(["x"] * 5))
    np.savez(tmp_path / "vis_ff.npz", y=np.array([0, 1]), g=np.array(["x", "y"]))
    assert th.finished(tmp_path, "vis") == {"ff"}
