#!/usr/bin/env python3
"""Independent check that the pretrained AASIST model loads and runs on THIS machine.

    python scripts/probe_aasist.py                       # load + synthetic-input smoke test
    python scripts/probe_aasist.py --media some_clip.mp4 # also score the clip's audio

This verifies loading, shapes, speed and determinism. It does NOT measure detection accuracy.
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from avdd.audio.aasist_detector import AasistDetector
from avdd.audio.extract import assess_audio, decode_audio
from avdd.audio.windows import WINDOW_SAMPLES, make_windows
from avdd.common.errors import AvddError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--variant", choices=["AASIST", "AASIST-L"], default="AASIST")
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ok = True
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]

        print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    except ImportError:
        print("FAIL: PyTorch is not installed (see docs/runbook_local.md)")
        return 1

    det = AasistDetector(args.variant, device=args.device)
    t = time.perf_counter()
    try:
        det.load()
    except AvddError as exc:
        print(f"FAIL load: {exc.code}: {exc.message}")
        return 1
    print(f"PASS load ({time.perf_counter() - t:.2f}s): {det.model_card.version}")

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(WINDOW_SAMPLES) * 0.05).astype(np.float32)
    t = time.perf_counter()
    s1 = det.score_windows([x])[0]
    dt = time.perf_counter() - t
    s2 = det.score_windows([x])[0]
    print(f"synthetic-noise window: score={s1:.4f} ({dt:.2f}s, first call incl. warm-up)")
    ok &= 0.0 <= s1 <= 1.0 and s1 == s2
    print(f"{'PASS' if ok else 'FAIL'} scores in [0,1] and deterministic (NOT a detection test)")

    if args.media:
        try:
            wave = decode_audio(args.media)
        except AvddError as exc:
            print(f"FAIL decode: {exc.code}: {exc.message}")
            return 1
        reasons, metrics = assess_audio(wave)
        print(f"decoded audio: {metrics}; usability gate reasons: {reasons or 'none'}")
        wins = make_windows(wave)
        t = time.perf_counter()
        scores = det.score_windows([w.samples for w in wins])
        print(f"scored {len(wins)} window(s) in {time.perf_counter() - t:.2f}s")
        for w, s in zip(wins, scores, strict=True):
            print(f"  {w.start_s:6.2f}-{w.end_s:6.2f}s  raw spoof score {s:.4f}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
