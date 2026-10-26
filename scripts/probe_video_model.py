#!/usr/bin/env python3
"""Independent check that the pretrained Hugging Face visual model loads and runs here.

    python scripts/probe_video_model.py --image face.jpg          # one face image (no cropping)
    python scripts/probe_video_model.py --video clip.mp4          # sample frames, crop faces, score
    python scripts/probe_video_model.py --image face.jpg --revision main   # if the pinned hash fails

Verifies download, label mapping, shapes and speed. It does NOT measure detection accuracy.
Needs: torch, transformers, pillow, opencv-python-headless and internet access to huggingface.co.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from avdd.common.errors import AvddError
from avdd.video.faces import HaarFaceDetector, crop_face_rgb
from avdd.video.frames import sample_frames
from avdd.video.hf_detector import DEFAULT_MODEL_ID, DEFAULT_REVISION, HfImageDetector


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path)
    g.add_argument("--video", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    det = HfImageDetector(args.model, args.revision, args.device)
    t = time.perf_counter()
    try:
        info = det.load()
    except AvddError as exc:
        print(f"FAIL load: {exc.code}: {exc.message}")
        return 1
    print(f"PASS load ({time.perf_counter() - t:.1f}s): {info}")

    faces: list[np.ndarray] = []
    if args.image:
        from PIL import Image

        faces = [np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)]
    else:
        fd = HaarFaceDetector()
        for fr in sample_frames(args.video, n_frames=8):
            boxes = fd.detect(fr.image_bgr)
            print(f"  t={fr.t_s:6.2f}s faces={len(boxes)}")
            if boxes:
                faces.append(crop_face_rgb(fr.image_bgr, boxes[0]))
    if not faces:
        print("FAIL: no face crops to score")
        return 1
    t = time.perf_counter()
    scores = det.score_faces(faces)
    print(f"scored {len(faces)} image(s) in {time.perf_counter() - t:.2f}s")
    for i, s in enumerate(scores):
        print(f"  image {i}: raw fake score {s:.4f}")
    ok = all(0.0 <= s <= 1.0 for s in scores)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
