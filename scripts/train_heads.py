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
import time
from collections import defaultdict
from collections.abc import Iterator
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


def visual_items(data: Path, fd: Any, frames_per_video: int, done: frozenset[str] = frozenset()) -> Iterator[tuple[str, Any, int, str]]:
    """Yield (source, face crop, label, group) one at a time, so memory stays flat."""
    import cv2
    import pyarrow.parquet as pq

    from avdd.video.frames import sample_frames

    for p in [] if "ff" in done else sorted((data / "ff").rglob("*.mp4")):
        label = int("/fake/" in str(p))
        group = "ff_" + (p.stem.split("__")[0].split("_")[0] if label else p.stem)
        for fr in sample_frames(p, frames_per_video):
            if (c := _faces(fr.image_bgr, fd)) is not None:
                yield "ff", c, label, group
    for tf in sorted((data / "WildDeepfake").rglob("*.tar.gz")):
        label = int("fake_" in tf.parent.name)
        part = "wdf_test" if tf.parent.name.endswith("_test") else "wdf"
        if part in done:
            continue
        with tarfile.open(tf) as t:
            seqs: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".png"):
                    seqs[str(Path(m.name).parent)].append(m)
            for seq, ms in list(seqs.items())[:40]:
                ms.sort(key=lambda m: m.name)
                for m in ms[:: max(1, len(ms) // 2)][:2]:
                    f = t.extractfile(m)
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
                    if img is not None:
                        yield part, _crop_or_whole(img, fd), label, f"wdf_{tf.stem}_{seq}"
    for pf in sorted((data / "celeb-df-image-dataset" / "data").glob("*.parquet")):
        part = "celeb_test" if pf.name.startswith("test") else "celeb"
        if part in done:
            continue
        keep_fake = 0.03 if part == "celeb_test" else 0.02  # all reals are kept; fakes outnumber them ~80:1
        for batch in pq.ParquetFile(pf).iter_batches(batch_size=256):
            for im, lab in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist(), strict=True):
                if lab == 0 and rng.random() > keep_fake:
                    continue
                img = cv2.imdecode(np.frombuffer(im["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    name = im["path"] or "x"
                    # celeb-df labels: 0 = Fake, 1 = Real
                    yield part, _crop_or_whole(img, fd), 1 - lab, "cdf_" + name.rsplit("_", 1)[0]


def audio_items(data: Path, per_source: dict[str, int], done: frozenset[str] = frozenset()) -> Iterator[tuple[str, Any, int, str]]:
    import pyarrow.parquet as pq
    import soundfile as sf

    from avdd.audio.windows import make_windows

    for name, cap in per_source.items():
        if name in done:
            continue
        shards = sorted((data / name / "data").glob("test-*.parquet"))
        rows = [(s, i) for s in shards for i in range(pq.ParquetFile(s).metadata.num_rows)]
        rng.shuffle(rows)
        by_shard: dict[Path, list[int]] = defaultdict(list)
        for s, i in rows[:cap]:
            by_shard[s].append(i)
        for s, idx in by_shard.items():
            pf = pq.ParquetFile(s)
            want, base = set(idx), 0
            for batch in pf.iter_batches(batch_size=128):
                for j, (a, y, n) in enumerate(zip(batch.column("audio").to_pylist(), batch.column("label").to_pylist(),
                                                   batch.column("notes").to_pylist(), strict=True)):  # fmt: skip
                    if base + j not in want:
                        continue
                    wave, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                    if sr != 16_000 or wave.ndim != 1 or len(wave) < 8000:
                        continue
                    notes = json.loads(n)
                    spk = notes.get("speaker") or notes.get("speaker_id") or "?"
                    yield name, make_windows(wave)[0].samples, int(y), f"{name}_{spk}"
                base += batch.num_rows


def finished(cache: Path, prefix: str) -> frozenset[str]:
    """Sources already embedded in an earlier (interrupted) run; they are skipped entirely."""
    return frozenset(p.stem.removeprefix(f"{prefix}_") for p in cache.glob(f"{prefix}_*.npz"))


def embed_stream(items: Iterator[tuple[str, Any, int, str]], det: Any, layers: tuple[int, ...], cache: Path, prefix: str) -> None:
    """Embed in batches of 32 as items arrive; save one .npz per source when it ends."""
    acc: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    buf: list[tuple[str, Any, int, str]] = []
    n, t0 = 0, time.time()

    def flush() -> None:
        nonlocal n
        if not buf:
            return
        feats = det.embed([x for _, x, _, _ in buf], layers)
        for i, (src, _, y, g) in enumerate(buf):
            acc[src]["y"].append(y)
            acc[src]["g"].append(g)
            for k in layers:
                acc[src][f"l{k}"].append(feats[k][i].astype(np.float16))
        n += len(buf)
        buf.clear()
        print(f"{prefix}: {n} embedded ({n / (time.time() - t0):.2f}/s)", flush=True)

    def save(src: str) -> None:
        d = acc.pop(src)
        np.savez(cache / f"{prefix}_{src}.npz", **{k: np.array(v) for k, v in d.items()})  # type: ignore[arg-type]
        print("saved", f"{prefix}_{src}", len(d["y"]), flush=True)

    current = None
    for item in items:
        if current is not None and item[0] != current:
            flush()
            save(current)
        current = item[0]
        if (cache / f"{prefix}_{current}.npz").exists():
            continue
        buf.append(item)
        if len(buf) >= 32:
            flush()
    flush()
    if current is not None and current in acc:
        save(current)


def embed_visual(data: Path, cache: Path, frames_per_video: int, device: str = "cpu") -> None:
    from avdd.video.clip_detector import ClipFaceDetector
    from avdd.video.faces import YuNetFaceDetector

    det = ClipFaceDetector(device=device, with_head=False)
    det.load()
    embed_stream(visual_items(data, YuNetFaceDetector(), frames_per_video, finished(cache, "vis")), det, VIS_LAYERS, cache, "vis")


def embed_audio(data: Path, cache: Path, caps: dict[str, int], device: str = "cpu") -> None:
    from avdd.audio.ssl_detector import SslAudioDetector

    det = SslAudioDetector(device=device, with_head=False)
    det.load()
    embed_stream(audio_items(data, caps, finished(cache, "aud")), det, AUD_LAYERS, cache, "aud")


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
    search: list[dict[str, float]] = []
    for layer in layers:
        xt, yt, _ = split(train, layer, "train")
        xv, yv, gv = split(train, layer, "val")
        mean, scale = xt.mean(0), xt.std(0) + 1e-6
        for c in (0.01, 0.1, 1.0):
            clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit((xt - mean) / scale, yt)
            lv = clf.decision_function((xv - mean) / scale)
            auc = _metrics(lv, yv)["auc"]
            print(f"{kind} layer {layer} C {c}: val auc {auc}", flush=True)
            search.append({"layer": layer, "C": c, "val_auc": auc})
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
    # Strictly above the (1 - fpr_target) quantile of genuine scores, so ties cannot flag more genuine media
    # than the target; never below 0.5, so nothing the calibrated model thinks is more likely real is flagged.
    t_fake = max(0.5, float(np.nextafter(np.quantile(gp[gy == 0], 1 - fpr_target), 1.0)))
    t_real = float(min(np.quantile(gp[gy == 0], 0.90), t_fake - 1e-3))
    report = {"layer": best["layer"], "C": best["C"], "t_fake": round(t_fake, 4), "t_real": round(t_real, 4),
              "validation": _metrics(gp, gy, t_fake, t_real), "tests": {}, "search": search,
              "n_train": len(xt)}  # fmt: skip
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
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--audio-caps", default="ASVspoof2019_LA=2000,ASVspoof5=2000,InTheWild=4500")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    if a.cmd == "embed-visual":
        embed_visual(a.data, a.cache, a.frames, a.device)
    elif a.cmd == "embed-audio":
        embed_audio(a.data, a.cache, {k: int(v) for k, v in (kv.split("=") for kv in a.audio_caps.split(","))}, a.device)
    else:
        fit(a.cache)


if __name__ == "__main__":
    main()
