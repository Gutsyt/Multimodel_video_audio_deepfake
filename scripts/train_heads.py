"""Train the calibrated linear heads on frozen CLIP / XLS-R embeddings.

  python scripts/train_heads.py embed-visual --data DATA --cache CACHE
  python scripts/train_heads.py embed-audio  --data DATA --cache CACHE
  python scripts/train_heads.py fit          --cache CACHE      # writes src/avdd/assets/*_head.npz

DATA holds the Hugging Face downloads listed in docs/MODEL_REPORT.md. Embeddings go through the
exact inference preprocessing (YuNet crop + CLIP processor; 16 kHz decode + 4 s window), so the
heads see at training time what they see in the app.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

VIS_LAYERS = (12, 16, 20, 24)
AUD_LAYERS = (4, 6, 8, 10, 12)
rng = random.Random(0)


def split_of(group: str) -> str:
    """Deterministic group-level split: 70% train, 15% val, 15% test."""
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


# ---------------------------------------------------------------- visual
def _faces(img_bgr: Any, fd: Any) -> Any:
    from avdd.video.faces import crop_face_rgb

    boxes = fd.detect(img_bgr)
    return crop_face_rgb(img_bgr, boxes[0]) if boxes else None


def _crop_or_whole(img_bgr: Any, fd: Any) -> Any:
    """Dataset images that are already tight face crops: re-crop if YuNet finds the face."""
    crop = _faces(img_bgr, fd)
    return crop if crop is not None else np.ascontiguousarray(img_bgr[:, :, ::-1])


def visual_items(data: Path, fd: Any, frames_per_video: int) -> dict[str, list[tuple[Any, int, str]]]:
    import cv2
    import pyarrow.parquet as pq

    from avdd.video.frames import sample_frames

    src: dict[str, list[tuple[Any, int, str]]] = defaultdict(list)
    for p in sorted((data / "ff").rglob("*.mp4")):
        label = int("/fake/" in str(p))
        stem = p.stem
        group = "ff_" + (stem.split("__")[0].split("_")[0] if label else stem)
        for fr in sample_frames(p, frames_per_video):
            if (c := _faces(fr.image_bgr, fd)) is not None:
                src["ff"].append((c, label, group))
    for tf in sorted((data / "WildDeepfake").rglob("*.tar.gz")):
        label = int("fake_" in tf.parent.name)
        part = "wdf_test" if tf.parent.name.endswith("_test") else "wdf"
        seqs: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
        with tarfile.open(tf) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".png"):
                    seqs[str(Path(m.name).parent)].append(m)
            for seq, ms in list(seqs.items())[:40]:
                ms.sort(key=lambda m: m.name)
                for m in ms[:: max(1, len(ms) // 2)][:2]:
                    f = t.extractfile(m)
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
                    if img is not None:
                        src[part].append((_crop_or_whole(img, fd), label, f"wdf_{tf.stem}_{seq}"))
    for pf in sorted((data / "celeb-df-image-dataset" / "data").glob("*.parquet")):
        part = "celeb_test" if pf.name.startswith("test") else "celeb"
        tbl = pq.read_table(pf)
        rows = list(range(tbl.num_rows))
        real = [i for i in rows if tbl.column("label")[i].as_py() == 1]
        fake = rng.sample([i for i in rows if tbl.column("label")[i].as_py() == 0], 3 * len(real) // 2 + 60)
        for i in real + fake:
            im = tbl.column("image")[i].as_py()
            img = cv2.imdecode(np.frombuffer(im["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                name = im["path"] or str(i)
                # celeb-df labels: 0 = Fake, 1 = Real
                src[part].append((_crop_or_whole(img, fd), 1 - tbl.column("label")[i].as_py(), "cdf_" + name.rsplit("_", 1)[0]))
    return src


def embed_visual(data: Path, cache: Path, frames_per_video: int) -> None:
    from avdd.video.clip_detector import ClipFaceDetector
    from avdd.video.faces import YuNetFaceDetector

    det = ClipFaceDetector(with_head=False)
    det.load()
    for name, items in visual_items(data, YuNetFaceDetector(), frames_per_video).items():
        out = cache / f"vis_{name}.npz"
        if out.exists():
            continue
        feats = det.embed([c for c, _, _ in items], VIS_LAYERS)
        np.savez(out, y=np.array([y for _, y, _ in items]), g=np.array([g for _, _, g in items]),
                 **{f"l{k}": v.astype(np.float16) for k, v in feats.items()})  # type: ignore[arg-type]  # fmt: skip
        print("saved", out.name, len(items), flush=True)


# ---------------------------------------------------------------- audio
def audio_items(data: Path, per_source: dict[str, int]) -> dict[str, list[tuple[Any, int, str]]]:
    import pyarrow.parquet as pq
    import soundfile as sf

    from avdd.audio.windows import make_windows

    src: dict[str, list[tuple[Any, int, str]]] = defaultdict(list)
    for name, cap in per_source.items():
        shards = sorted((data / name / "data").glob("test-*.parquet"))
        rows = [(s, i) for s in shards for i in range(pq.ParquetFile(s).metadata.num_rows)]
        rng.shuffle(rows)
        by_shard: dict[Path, list[int]] = defaultdict(list)
        for s, i in rows[:cap]:
            by_shard[s].append(i)
        for s, idx in by_shard.items():
            tbl = pq.read_table(s).take(idx)
            for a, y, n in zip(tbl.column("audio").to_pylist(), tbl.column("label").to_pylist(),
                               tbl.column("notes").to_pylist(), strict=True):  # fmt: skip
                wave, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                if sr != 16_000 or wave.ndim != 1 or len(wave) < 8000:
                    continue
                notes = json.loads(n)
                spk = notes.get("speaker") or notes.get("speaker_id") or "?"
                src[name].append((make_windows(wave)[0].samples, int(y), f"{name}_{spk}"))
    return src


def embed_audio(data: Path, cache: Path, caps: dict[str, int]) -> None:
    from avdd.audio.ssl_detector import SslAudioDetector

    det = SslAudioDetector(with_head=False)
    det.load()
    for name, items in audio_items(data, caps).items():
        out = cache / f"aud_{name}.npz"
        if out.exists():
            continue
        feats = det.embed([w for w, _, _ in items], AUD_LAYERS)
        np.savez(out, y=np.array([y for _, y, _ in items]), g=np.array([g for _, _, g in items]),
                 **{f"l{k}": v.astype(np.float16) for k, v in feats.items()})  # type: ignore[arg-type]  # fmt: skip
        print("saved", out.name, len(items), flush=True)


# ---------------------------------------------------------------- fit
def _load(cache: Path, names: list[str], layer: int) -> tuple[Any, Any, Any]:
    zs = [np.load(cache / f"{n}.npz") for n in names]
    x = np.concatenate([z[f"l{layer}"] for z in zs]).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True), np.concatenate([z["y"] for z in zs]), np.concatenate([z["g"] for z in zs])


def _group_scores(p: Any, y: Any, g: Any) -> tuple[Any, Any]:
    """Median per group (video / sequence / speaker-clip): what the app's decision sees."""
    keys = sorted(set(g))
    idx = {k: np.where(g == k)[0] for k in keys}
    return np.array([np.median(p[idx[k]]) for k in keys]), np.array([y[idx[k][0]] for k in keys])


def _metrics(p: Any, y: Any, t_fake: float | None = None, t_real: float | None = None) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    eer = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
    m = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "auc": round(float(roc_auc_score(y, p)), 4), "eer": round(eer, 4)}
    if t_fake is not None and t_real is not None:
        real, fake = p[y == 0], p[y == 1]
        m |= {"genuine_flagged": round(float((real >= t_fake).mean()), 4),
              "genuine_cleared": round(float((real <= t_real).mean()), 4),
              "fake_flagged": round(float((fake >= t_fake).mean()), 4),
              "fake_cleared": round(float((fake <= t_real).mean()), 4)}  # fmt: skip
    return m


def fit_head(cache: Path, kind: str, train: list[str], tests: dict[str, list[str]], layers: tuple[int, ...],
             fpr_target: float, meta: dict[str, Any], grouped: bool) -> dict[str, Any]:  # fmt: skip
    from sklearn.linear_model import LogisticRegression

    def split(names: list[str], layer: int, which: str) -> tuple[Any, Any, Any]:
        x, y, g = _load(cache, names, layer)
        keep = np.array([split_of(k) == which for k in g])
        return x[keep], y[keep], g[keep]

    best: dict[str, Any] = {}
    for layer in layers:
        xt, yt, _ = split(train, layer, "train")
        xv, yv, gv = split(train, layer, "val")
        mean, scale = xt.mean(0), xt.std(0) + 1e-6
        for c in (0.01, 0.1, 1.0):
            clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit((xt - mean) / scale, yt)
            lv = clf.decision_function((xv - mean) / scale)
            auc = _metrics(lv, yv)["auc"]
            print(f"{kind} layer {layer} C {c}: val auc {auc}", flush=True)
            if auc > best.get("auc", 0):
                best = {"auc": auc, "layer": layer, "C": c, "clf": clf, "mean": mean, "scale": scale, "lv": lv, "yv": yv, "gv": gv}

    # Platt scaling on validation logits, then thresholds on validation group medians.
    cal = LogisticRegression(C=1e4).fit(best["lv"][:, None], best["yv"])
    a, b = float(cal.coef_[0, 0]), float(cal.intercept_[0])

    def prob(x: Any) -> Any:
        logit = ((x - best["mean"]) / best["scale"]) @ best["clf"].coef_[0] + best["clf"].intercept_[0]
        return 1 / (1 + np.exp(-(a * logit + b)))

    pv = 1 / (1 + np.exp(-(a * best["lv"] + b)))
    gp, gy = _group_scores(pv, best["yv"], best["gv"]) if grouped else (pv, best["yv"])
    t_fake = float(np.quantile(gp[gy == 0], 1 - fpr_target))  # at most fpr_target of genuine flagged
    t_real = float(min(np.quantile(gp[gy == 0], 0.90), t_fake - 1e-3))
    report = {"layer": best["layer"], "C": best["C"], "t_fake": round(t_fake, 4), "t_real": round(t_real, 4),
              "validation": _metrics(gp, gy, t_fake, t_real), "tests": {}}  # fmt: skip
    for tname, names in tests.items():
        x, y, g = _load(cache, names, best["layer"])
        if tname == "in_domain_test":
            keep = np.array([split_of(k) == "test" for k in g])
            x, y, g = x[keep], y[keep], g[keep]
        p = prob(x)
        gp, gy = _group_scores(p, y, g) if grouped else (p, y)
        report["tests"][tname] = _metrics(gp, gy, t_fake, t_real)
    meta = {**meta, **{k: report[k] for k in ("layer", "t_fake", "t_real")}, "metrics": report,
            "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d")}  # fmt: skip
    from avdd.common.head import ASSETS

    np.savez(ASSETS / f"{kind}_head.npz", mean=best["mean"], scale=best["scale"], w=best["clf"].coef_[0],
             b=best["clf"].intercept_[0], cal_a=a, cal_b=b, meta=json.dumps(meta))  # fmt: skip
    return report


def fit(cache: Path) -> None:
    vis = fit_head(
        cache, "visual", ["vis_ff", "vis_wdf", "vis_celeb"],
        {"in_domain_test": ["vis_ff", "vis_wdf", "vis_celeb"], "wilddeepfake_test": ["vis_wdf_test"], "celebdf_test": ["vis_celeb_test"]},
        VIS_LAYERS, 0.02, grouped=True,
        meta={"trained_on": "FaceForensics++ c23 (real + 6 methods), Celeb-DF/FakeAVCeleb frames, WildDeepfake train",
              "limitations": ["Image-level probe on face crops: no temporal or lip-sync modelling.",
                              "Unseen manipulation methods and heavy filters can still evade it or raise false alarms.",
                              "Thresholds were chosen on validation data for at most 2% genuine videos flagged."]},
    )  # fmt: skip
    print(json.dumps(vis, indent=1))
    aud = fit_head(
        cache, "audio", ["aud_ASVspoof2019_LA", "aud_ASVspoof5", "aud_InTheWild"],
        {"in_domain_test": ["aud_ASVspoof2019_LA", "aud_ASVspoof5", "aud_InTheWild"]},
        AUD_LAYERS, 0.02, grouped=False,
        meta={"trained_on": "ASVspoof 2019 LA, ASVspoof 5 (with codecs), In-the-Wild (speaker-disjoint split)",
              "limitations": ["Judges speech only; music or noise is marked unreliable by the input gate.",
                              "New voice-cloning systems may evade it.",
                              "Thresholds were chosen on validation data for at most 2% genuine clips flagged."]},
    )  # fmt: skip
    print(json.dumps(aud, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed-visual", "embed-audio", "fit"])
    ap.add_argument("--data", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--audio-caps", default="ASVspoof2019_LA=3000,ASVspoof5=3000,InTheWild=7000")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    if a.cmd == "embed-visual":
        embed_visual(a.data, a.cache, a.frames)
    elif a.cmd == "embed-audio":
        embed_audio(a.data, a.cache, {k: int(v) for k, v in (kv.split("=") for kv in a.audio_caps.split(","))})
    else:
        fit(a.cache)


if __name__ == "__main__":
    main()
