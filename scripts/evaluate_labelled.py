#!/usr/bin/env python3
"""Measure the real branches on LABELLED samples you provide. This is the only way to learn a false-positive
rate: nothing else in this repository can.

    python scripts/evaluate_labelled.py --real-audio .\\real_wavs --real-faces .\\real_faces --fake-faces .\\fake_faces

Directories hold audio (any ffmpeg-readable format) or images. Audio needs real clips only (specificity is
reported); faces can have real and fake (specificity, sensitivity and AUC). Results describe YOUR samples only.
The 0.5 threshold is the application's unvalidated default; AUC does not depend on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from avdd.audio.aasist_detector import AasistDetector
from avdd.audio.branch import run_audio_branch
from avdd.video.branch import run_image_branch
from avdd.video.faces import HaarFaceDetector
from avdd.video.hf_detector import DEFAULT_MODEL_ID, DEFAULT_REVISION, HfImageDetector

THRESHOLD = 0.5


def files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file())


def auc(pos: list[float], neg: list[float]) -> float:
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-audio", type=Path)
    ap.add_argument("--real-faces", type=Path)
    ap.add_argument("--fake-faces", type=Path)
    a = ap.parse_args()
    if a.real_audio:
        det = AasistDetector("AASIST")
        det.load()
        sc = [o.evidence.score for f in files(a.real_audio) if (o := run_audio_branch(f, det)).evidence.usable and o.evidence.score is not None]
        print(f"audio, {len(sc)} real clips: mean {np.mean(sc):.3f}, reached {THRESHOLD}: {sum(s >= THRESHOLD for s in sc)}/{len(sc)} (specificity only)")
    if a.real_faces:
        vis, haar = HfImageDetector(DEFAULT_MODEL_ID, DEFAULT_REVISION), HaarFaceDetector()
        vis.load()

        def scores(d: Path) -> list[float]:
            out = [run_image_branch(f, vis, haar) for f in files(d)]
            return [o.evidence.score for o in out if o.evidence.usable and o.evidence.score is not None]

        real = scores(a.real_faces)
        print(f"faces, {len(real)} real: mean {np.mean(real):.3f}, reached {THRESHOLD}: {sum(s >= THRESHOLD for s in real)}/{len(real)}")
        if a.fake_faces:
            fake = scores(a.fake_faces)
            print(f"faces, {len(fake)} fake: mean {np.mean(fake):.3f}, reached {THRESHOLD}: {sum(s >= THRESHOLD for s in fake)}/{len(fake)}")
            print(f"AUC (fake ranked above real): {auc(fake, real):.3f}   0.5 = chance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
