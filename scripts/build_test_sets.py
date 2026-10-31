"""Build end-to-end test FILES from the held-out (test-split) part of the data, for evaluate_folder.py.

    python scripts/build_test_sets.py DATA OUT

OUT/A_ff/{real,fake}   60 + 60 FaceForensics++ videos (test-split identities only)
OUT/B_wdf/{real,fake}  WildDeepfake test sequences re-encoded as 25 fps H.264 mp4
OUT/C_itw/{real,fake}  80 + 80 In-the-Wild clips from test-split speakers
OUT/D_av/real          40 genuine videos WITH genuine speech (A_ff real video + C_itw real audio);
                       checks the combined false-alarm rate when both checks run
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from train_heads import split_of

rng = random.Random(1)


def ff_group(p: Path) -> str:
    """Identity group exactly as train_heads.visual_items assigns it."""
    return "ff_" + (p.stem.split("__")[0].split("_")[0] if "/fake/" in str(p) else p.stem)


def ff(cmd: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *cmd], check=True)


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    for lab in ("real", "fake"):
        vids = [p for p in sorted((data / "ff").rglob("*.mp4")) if ("/fake/" in str(p)) == (lab == "fake")]
        d = out / "A_ff" / lab
        d.mkdir(parents=True, exist_ok=True)
        for p in rng.sample([p for p in vids if split_of(ff_group(p)) == "test"], 60):
            (d / f"{p.parent.name}_{p.name}").write_bytes(p.read_bytes())

    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for lab in ("real", "fake"):
        d, n = out / "B_wdf" / lab, 0
        d.mkdir(parents=True, exist_ok=True)
        for tf in sorted((data / "WildDeepfake" / "deepfake_in_the_wild" / f"{lab}_test").glob("*.tar.gz")):
            with tarfile.open(tf) as t:
                seqs: dict[str, list[tarfile.TarInfo]] = {}
                for m in t.getmembers():
                    if m.isfile() and m.name.endswith(".png"):
                        seqs.setdefault(str(Path(m.name).parent), []).append(m)
                for seq, ms in list(seqs.items())[:4]:
                    if len(ms) < 20:
                        continue
                    ms.sort(key=lambda m: int(Path(m.name).stem))
                    for i, m in enumerate(ms[:75]):
                        f = t.extractfile(m)
                        if f:
                            (tmp / f"{i:04d}.png").write_bytes(f.read())
                    ff(["-framerate", "25", "-i", str(tmp / "%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(d / f"{tf.stem}_{Path(seq).name}.mp4")])
                    n += 1
                    for f2 in tmp.iterdir():
                        f2.unlink()
            if n >= 40:
                break
    tmp.rmdir()

    rows = []
    for s in sorted((data / "InTheWild" / "data").glob("test-*.parquet")):
        t = pq.read_table(s, columns=["notes", "label"])
        rows += [(s, i, json.loads(n)["speaker"], y) for i, (n, y) in
                 enumerate(zip(t.column("notes").to_pylist(), t.column("label").to_pylist(), strict=True))]
    test = [r for r in rows if split_of(f"InTheWild_{r[2]}") == "test"]
    print("In-the-Wild test speakers:", sorted({r[2] for r in test}))
    for lab, y in (("real", 0), ("fake", 1)):
        d = out / "C_itw" / lab
        d.mkdir(parents=True, exist_ok=True)
        for s, i, spk, _ in rng.sample([r for r in test if r[3] == y], 80):
            a = pq.read_table(s, columns=["audio"]).slice(i, 1).column("audio")[0].as_py()
            (d / f"{spk.replace(' ', '_')}_{i}.flac").write_bytes(a["bytes"])

    d = out / "D_av" / "real"
    d.mkdir(parents=True, exist_ok=True)
    vids, auds = sorted((out / "A_ff" / "real").glob("*.mp4")), sorted((out / "C_itw" / "real").glob("*.flac"))
    for k, (v, a) in enumerate(zip(vids[:40], auds[:40], strict=False)):
        ff(["-i", str(v), "-stream_loop", "-1", "-i", str(a), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-c:a", "aac", "-shortest", str(d / f"av_{k}.mp4")])
    print({p.name: {q.name: len(list(q.iterdir())) for q in p.iterdir()} for p in out.iterdir() if p.is_dir()})


if __name__ == "__main__":
    main()
