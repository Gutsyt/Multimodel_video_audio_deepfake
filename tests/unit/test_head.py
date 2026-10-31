"""LinearHead metadata must stay valid JSON for the API even when a metric could not be computed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avdd.common.head import LinearHead


def test_nan_metrics_load_as_none(tmp_path: Path) -> None:
    meta = {"layer": 16, "t_fake": 0.7, "t_real": 0.3, "metrics": {"tests": {"x": {"auc": float("nan"), "eer": float("nan")}}}}
    p = tmp_path / "h.npz"
    np.savez(p, mean=np.zeros(4), scale=np.ones(4), w=np.ones(4), b=0.0, cal_a=1.0, cal_b=0.0, meta=json.dumps(meta))
    head = LinearHead(p)
    assert head.meta["metrics"]["tests"]["x"] == {"auc": None, "eer": None}
    json.dumps(head.meta, allow_nan=False)  # what the API's JSON encoder requires
    assert head.t_fake == 0.7
