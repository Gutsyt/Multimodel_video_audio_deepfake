#!/usr/bin/env python3
"""Analyse one video file with the pretrained detectors and print a JSON result.

Run from the repository root (venv active):

    python scripts/analyze_video.py path/to/clip.mp4 --out result.json
    python scripts/analyze_video.py clip.mp4 --audio AASIST --video-model none     # audio only
    python scripts/analyze_video.py clip.mp4 --audio none --video-model hf         # visual only

If a model cannot be loaded the corresponding modality is reported as NOT_ANALYZED (never as
authentic). Exit codes: 0 ok, 2 input/environment error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from avdd.common.errors import AvddError, ModelLoadError
from avdd.common.protocols import AudioWindowDetector, FaceImageDetector
from avdd.pipeline.analyze import AnalysisConfig, analyze_media
from avdd.video.branch import VideoBranchConfig
from avdd.video.faces import HaarFaceDetector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write JSON here (default: stdout)")
    ap.add_argument("--audio", choices=["AASIST", "AASIST-L", "none"], default="AASIST")
    ap.add_argument("--video-model", choices=["hf", "none"], default="hf")
    ap.add_argument("--hf-model", default=None, help="override Hugging Face model id")
    ap.add_argument("--revision", default=None, help="Hugging Face revision (default: pinned)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    audio: AudioWindowDetector | None = None
    video: FaceImageDetector | None = None
    if args.audio != "none":
        from avdd.audio.aasist_detector import AasistDetector

        a = AasistDetector(args.audio, device=args.device)
        try:
            a.load()
            audio = a
        except ModelLoadError as exc:
            print(f"warning: audio model not loaded ({exc.code}): audio will be NOT_ANALYZED", file=sys.stderr)
    if args.video_model == "hf":
        from avdd.video.hf_detector import DEFAULT_MODEL_ID, DEFAULT_REVISION, HfImageDetector

        v = HfImageDetector(args.hf_model or DEFAULT_MODEL_ID, args.revision or DEFAULT_REVISION, args.device)
        try:
            print(f"loaded visual model: {v.load()}", file=sys.stderr)
            video = v
        except ModelLoadError as exc:
            print(f"warning: visual model not loaded ({exc.code}: {exc.message}): visual will be NOT_ANALYZED", file=sys.stderr)

    cfg = AnalysisConfig(max_seconds=args.max_seconds,
                         video=VideoBranchConfig(n_frames=args.frames, max_seconds=args.max_seconds))  # fmt: skip
    try:
        result = analyze_media(args.video, audio, video, HaarFaceDetector(), cfg)
    except AvddError as exc:
        print(f"error ({exc.code}): {exc.message}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
