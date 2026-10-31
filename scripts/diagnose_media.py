#!/usr/bin/env python3
"""Run the real pipeline on ONE file and print the verdict with per-frame / per-window scores.

    python scripts/diagnose_media.py clip.mp4 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avdd.audio.ssl_detector import SslAudioDetector
from avdd.pipeline.analyze import analyze_media
from avdd.video.clip_detector import ClipFaceDetector
from avdd.video.faces import YuNetFaceDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    audio, visual = SslAudioDetector(), ClipFaceDetector()
    audio.load()
    visual.load()
    r = analyze_media(args.path, audio, visual, YuNetFaceDetector())
    a = r["assessment"]
    print(f"{a['verdict'].upper()}  ({a['basis']})\n{a['summary']}")
    for name, m in r["modalities"].items():
        print(f"\n[{name}] {m['status']} score={m['score']} verdict={m.get('verdict')} {m['reasons']}")
        for f in m.get("frames") or []:
            print(f"  frame {f['frame_index']:>5} t={f['t_s']} faces={f['n_faces']} score={f.get('score')}")
        for s in m.get("segments") or []:
            print(f"  {s['start_s']:6.1f}-{s['end_s']:6.1f}s score={s['score']}")
    if args.json:
        args.json.write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
