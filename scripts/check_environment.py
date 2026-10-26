#!/usr/bin/env python3
"""Report whether this machine can run the project. Run from the repository root:

    python scripts/check_environment.py [--json] [--path /dataset/disk]

Exit code 0 if hard requirements are met (Python >= 3.10, ffmpeg and ffprobe on
PATH); 1 otherwise. Resource findings (RAM, disk, GPU) are reported as facts and
warnings, never as failures.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MIN_PYTHON = (3, 10)


def _first_line(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.stdout.splitlines()
    return lines[0] if lines else None


def _mem_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return None


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return {"installed": False}
    return {"installed": True, "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available())}


def collect(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path if path.exists() else Path("."))
    py_ok = sys.version_info[:2] >= MIN_PYTHON
    ffmpeg, ffprobe = _first_line(["ffmpeg", "-version"]), _first_line(["ffprobe", "-version"])
    warnings: list[str] = []
    if (mem := _mem_gb()) is not None and mem < 8:
        warnings.append(f"RAM {mem} GB is low for model training; use cached features or Colab/Kaggle.")
    if usage.free / 1e9 < 100:
        warnings.append(f"Free disk {usage.free / 1e9:.0f} GB at {path}: full dataset + crops may not fit.")
    torch_info = _torch_info()
    if not torch_info.get("cuda_available"):
        warnings.append("No CUDA GPU visible: training will be very slow; inference only on CPU.")
    return {
        "python": platform.python_version(),
        "python_ok": py_ok,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _mem_gb(),
        "disk_free_gb": round(usage.free / 1e9, 1),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "torch": torch_info,
        "warnings": warnings,
        "ok": bool(py_ok and ffmpeg and ffprobe),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--path", type=Path, default=Path("."), help="path whose disk space to report")
    args = ap.parse_args()
    info = collect(args.path)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:14}: {v}")
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
