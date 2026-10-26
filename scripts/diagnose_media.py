#!/usr/bin/env python3
"""Explain what each branch did with ONE media file (video, audio or image).

    python scripts/diagnose_media.py clip.mp4
    python scripts/diagnose_media.py clip.mp4 --sensitivity --json report.json

Runs the same pipeline as the server with the real pretrained models and prints, per branch: the
score, the raw logits of both classes, per-window / per-frame values, input-quality metrics and the
verified output semantics.

--sensitivity re-scores the same media with small, documented preprocessing changes (the other AASIST
variant, loudness-normalised audio, a shifted window grid, different face-crop margins, the whole
frame instead of a crop). These probes show how FRAGILE a score is. They are NOT accuracy
measurements: without labelled ground truth nothing here can say whether a flag is a false positive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from avdd.audio.aasist_detector import AasistDetector
from avdd.audio.extract import decode_audio
from avdd.audio.windows import make_windows
from avdd.common.errors import AvddError
from avdd.media.probe import probe_media
from avdd.pipeline.analyze import analyze_media
from avdd.video.faces import HaarFaceDetector, crop_face_rgb
from avdd.video.frames import SampledFrame, load_image_frame, sample_frames
from avdd.video.hf_detector import DEFAULT_MODEL_ID, DEFAULT_REVISION, HfImageDetector


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def audio_sensitivity(path: Path, variant: str, max_seconds: float) -> dict[str, Any]:
    wave = decode_audio(path, max_seconds=max_seconds)
    out: dict[str, Any] = {}

    def mean_score(det: AasistDetector, w: NDArray[np.float32]) -> float | None:
        wins = make_windows(w)
        return _mean(det.score_windows([x.samples for x in wins])) if wins else None

    base = AasistDetector(variant)
    base.load()
    out[f"{variant} (as served)"] = mean_score(base, wave)
    peak = float(np.max(np.abs(wave))) or 1.0
    out[f"{variant}, peak-normalised to 0.9"] = mean_score(base, (wave / peak * 0.9).astype(np.float32))
    shift = 2 * 16_000
    if wave.shape[0] > shift + 16_000:
        out[f"{variant}, window grid shifted by 2 s"] = mean_score(base, wave[shift:])
    other = "AASIST-L" if variant == "AASIST" else "AASIST"
    alt = AasistDetector(other)
    alt.load()
    out[f"{other} (other checkpoint)"] = mean_score(alt, wave)
    return out


def _frames_for(path: Path, n_frames: int, max_seconds: float) -> list[SampledFrame]:
    if probe_media(path).kind == "image":
        return [load_image_frame(path)]
    return sample_frames(path, n_frames, max_seconds)


def visual_sensitivity(path: Path, n_frames: int, max_seconds: float) -> dict[str, Any]:
    det = HfImageDetector(DEFAULT_MODEL_ID, DEFAULT_REVISION)
    det.load()
    haar = HaarFaceDetector()
    frames = _frames_for(path, n_frames, max_seconds)
    out: dict[str, Any] = {}
    for margin in (0.1, 0.3, 0.5):
        crops = []
        for fr in frames:
            faces = haar.detect(fr.image_bgr)
            if faces:
                crops.append(crop_face_rgb(fr.image_bgr, faces[0], margin))
        out[f"face crop, margin {margin:g}" + (" (as served)" if margin == 0.3 else "")] = (
            _mean(det.score_faces(crops)) if crops else None
        )
    full = [np.ascontiguousarray(fr.image_bgr[:, :, ::-1]) for fr in frames]
    out["whole frame, no face crop"] = _mean(det.score_faces(full))
    return out


def print_summary(result: dict[str, Any]) -> None:
    asm = result.get("assessment") or {}
    print(f"\nASSESSMENT  {asm.get('status')}  (basis: {asm.get('basis')}, threshold {asm.get('threshold')} UNVALIDATED)")
    print(f"            {asm.get('summary')}")
    for name, m in result["modalities"].items():
        print(f"\n== {name.upper()}  status={m['status']}  score={m['score']}  decision={m.get('decision')}")
        for r in m["reasons"]:
            print(f"   reason: {r}")
        d = m.get("diagnostics") or {}
        sem = d.get("output_semantics")
        if sem:
            print(f"   score is softmax mass on '{sem['score_class']}' (index {sem['score_class_index']}); "
                  f"classes {sem['class_labels']}")
        for key in ("input", "windows", "scores"):
            if key in d:
                print(f"   {key}: {json.dumps(d[key])}")
        for seg in m.get("segments") or []:
            print(f"   window {seg['start_s']:>6.1f}-{seg['end_s']:>6.1f}s  score={seg['score']:.4f}  logits={seg.get('logits')}")
        for fr in m.get("frames") or []:
            if "score" in fr:
                print(f"   frame {fr['frame_index']:>5} t={fr['t_s']}  score={fr['score']:.4f}  "
                      f"face_area={fr.get('face_area_ratio')} crop={fr.get('crop_px')} logits={fr.get('logits')}")
            else:
                print(f"   frame {fr['frame_index']:>5} t={fr['t_s']}  no face (detections: {fr['n_faces']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--audio-model", default="AASIST", choices=["AASIST", "AASIST-L"])
    ap.add_argument("--sensitivity", action="store_true", help="re-score with documented preprocessing changes")
    ap.add_argument("--json", type=Path, help="write the full result (and sensitivity) as JSON")
    args = ap.parse_args()
    if not args.path.is_file():
        print("file not found", file=sys.stderr)
        return 2

    audio = AasistDetector(args.audio_model)
    visual = HfImageDetector(DEFAULT_MODEL_ID, DEFAULT_REVISION)
    try:
        audio.load()
        info = visual.load()
        print(f"visual model labels: {info['id2label']}  fake_index={info['fake_index']}")
        result = analyze_media(args.path, audio, visual, HaarFaceDetector())
    except AvddError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    print_summary(result)

    report: dict[str, Any] = {"result": result}
    if args.sensitivity:
        sens: dict[str, Any] = {"note": "fragility probes, not accuracy measurements"}
        if result["modalities"]["audio"]["status"] == "available":
            sens["audio_mean_score"] = audio_sensitivity(args.path, args.audio_model, 60.0)
        if result["modalities"]["visual"]["status"] == "available":
            sens["visual_mean_score"] = visual_sensitivity(args.path, 16, 60.0)
        report["sensitivity"] = sens
        print("\nSENSITIVITY (same media, changed preprocessing; spread shows fragility, not correctness)")
        for branch, probes in sens.items():
            if isinstance(probes, dict):
                vals = [v for v in probes.values() if isinstance(v, float)]
                spread = f"  spread={max(vals) - min(vals):.3f}" if len(vals) > 1 else ""
                print(f" {branch}:{spread}")
                for k, v in probes.items():
                    print(f"   {k:<42} {v}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
