"""Inventory tool tests on a SYNTHETIC tree (software behaviour only)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from avdd.data.inventory import InspectionConfig, inspect_table, run_inspection
from tests.conftest import requires_ffmpeg


def _all_keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _all_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _all_keys(v)}
    return set()


@pytest.fixture()
def tree(media_dir: Path, tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "groupA").mkdir(parents=True)
    (root / "groupB").mkdir()
    shutil.copy(media_dir / "video_audio.mp4", root / "groupA" / "a.mp4")
    shutil.copy(media_dir / "video_audio.mp4", root / "groupA" / "a_copy.mp4")  # exact duplicate
    shutil.copy(media_dir / "video_only.mp4", root / "groupA" / "fake_looking_name.mp4")
    shutil.copy(media_dir / "audio_only.wav", root / "groupB" / "c.wav")
    shutil.copy(media_dir / "garbage.mp4", root / "groupB" / "bad.mp4")
    (root / "meta.csv").write_text("clip,tag,note\na.mp4,x,hello\nc.wav,y,\nbad.mp4,x,world\n")
    (root / "cfg.json").write_text(json.dumps({"k": 1, "j": 2}))
    return root


@requires_ffmpeg
def test_inventory_reports_observed_facts(tree: Path, tmp_path: Path) -> None:
    inv = run_inspection(InspectionConfig(root=tree, out_dir=tmp_path / "out", hash_media=True))
    assert inv["totals"]["n_files"] == 7
    assert inv["totals"]["n_media_files"] == 5
    s = inv["media_probe"]["summary"]
    assert s["status_counts"] == {"ok": 4, "failed": 1}
    assert s["stream_combinations"] == {"video+audio": 2, "video_only": 1, "audio_only": 1,
                                        "unprobeable:failed": 1}  # fmt: skip
    assert s["stream_combinations_by_top_dir"]["groupB"] == {"audio_only": 1, "unprobeable:failed": 1}
    assert inv["duplicates"]["n_groups"] == 1
    assert (tmp_path / "out" / "inventory.json").is_file()
    assert (tmp_path / "out" / "INVENTORY.md").is_file()
    lines = (tmp_path / "out" / "media_probe.jsonl").read_text().splitlines()
    assert len(lines) == 5


@requires_ffmpeg
def test_inventory_never_infers_labels_from_names(tree: Path, tmp_path: Path) -> None:
    inv = run_inspection(InspectionConfig(root=tree, out_dir=tmp_path / "out"))
    keys = {k.lower() for k in _all_keys(inv)}
    assert not any(w in keys for w in ("label", "labels", "split", "is_fake", "manipulation", "identity"))


def test_table_summary_counts_columns_empties_and_values(tree: Path) -> None:
    info = inspect_table(tree / "meta.csv", ",")
    assert info["columns"] == ["clip", "tag", "note"] and info["n_rows"] == 3
    assert info["column_info"]["note"]["empty_cells"] == 1
    assert info["column_info"]["tag"]["value_counts"] == {"x": 2, "y": 1}


@requires_ffmpeg
def test_sampling_is_deterministic_and_flagged(tree: Path, tmp_path: Path) -> None:
    def run(out: str) -> list[str]:
        cfg = InspectionConfig(root=tree, out_dir=tmp_path / out, max_probe=2, seed=7)
        inv = run_inspection(cfg)
        assert inv["media_probe"]["sampled"] is True
        return [json.loads(x)["path"] for x in (tmp_path / out / "media_probe.jsonl").read_text().splitlines()]

    first, second = run("o1"), run("o2")
    assert first == second and len(first) == 2


def test_no_probe_mode_does_not_need_ffmpeg(tree: Path, tmp_path: Path) -> None:
    inv = run_inspection(InspectionConfig(root=tree, out_dir=tmp_path / "out", probe=False))
    assert inv["media_probe"]["summary"] is None and inv["totals"]["n_media_files"] == 5


def test_non_directory_root_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        run_inspection(InspectionConfig(root=tmp_path / "nope", out_dir=tmp_path / "out"))
