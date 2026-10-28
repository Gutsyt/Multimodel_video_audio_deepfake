#!/usr/bin/env python3
"""Run the full app pipeline over folders of KNOWN genuine and/or fake files; print verdict rates.

    python scripts/evaluate_folder.py --real my_real_videos/ --fake my_fake_videos/ [--json out.json]

Use it to check the false-alarm rate on media like yours (phone videos, calls, ...).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from avdd.audio.ssl_detector import SslAudioDetector
from avdd.pipeline.analyze import analyze_media
from avdd.video.clip_detector import ClipFaceDetector
from avdd.video.faces import YuNetFaceDetector

EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path)
    ap.add_argument("--fake", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    audio, visual, faces = SslAudioDetector(), ClipFaceDetector(), YuNetFaceDetector()
    audio.load()
    visual.load()
    rows = []
    for label, folder in (("real", a.real), ("fake", a.fake)):
        for p in sorted(folder.rglob("*")) if folder else []:
            if p.suffix.lower() in EXT:
                r = analyze_media(p, audio, visual, faces)["assessment"]
                rows.append({"file": p.name, "label": label, "verdict": r["verdict"], "score": r["score"]})
                print(f"{label:4} {r['verdict']:22} {r['score'] if r['score'] is None else round(r['score'], 3)!s:6} {p.name}", flush=True)
    for label in ("real", "fake"):
        c = Counter(r["verdict"] for r in rows if r["label"] == label)
        n = sum(c.values())
        if n:
            print(f"\n{label}: n={n} " + ", ".join(f"{k}={v} ({100 * v / n:.1f}%)" for k, v in c.most_common()))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
