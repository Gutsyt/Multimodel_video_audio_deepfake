"""Download the training / test data used by train_heads.py (about 10 GB) into DATA.

    python scripts/download_data.py DATA [--ff-per-method 170] [--wdf-tars 12]

Nothing is extracted or committed. FaceForensics++ ships as one 18 GB zip; only the chosen videos are
read out of it with HTTP range requests. Re-running skips files that are already there.
Set HF_TOKEN for faster, rate-limit-free downloads (optional).
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (lets zipfile read one member of a remote zip)."""

    def __init__(self, url: str) -> None:
        self.c = httpx.Client(follow_redirects=True, timeout=120)
        r = self.c.head(url)
        self.url, self.size, self.pos = str(r.url), int(r.headers["Content-Length"]), 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                data = self.c.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}).content
                break
            except httpx.TransportError:
                if attempt == 4:
                    raise
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def get(repo: str, files: list[str], out: Path) -> None:
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=out / repo.split("/")[-1])
        print("ok", repo, f, flush=True)


def parquet(repo: str) -> list[str]:
    return [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]


def faceforensics(out: Path, per_method: int) -> None:
    repo, name = "bitmind/FaceForensicsC23", "FaceForensics++_C23.zip"

    def open_zip() -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BufferedReader(HttpFile(hf_hub_url(repo, name, repo_type="dataset")), 1 << 20))  # type: ignore[type-var]

    names = [i.filename for i in open_zip().infolist() if i.filename.endswith(".mp4")]
    pick = [n for n in names if "/real/" in n]
    for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"):
        fm = sorted(n for n in names if f"/fake/{m}/" in n)
        pick += fm[:: max(1, len(fm) // per_method)][:per_method]

    def job(chunk: list[str]) -> int:
        z = open_zip()
        for n in chunk:
            dst = out / "ff" / n.split("C23/")[1]
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(f"{dst}.part", "wb") as f:
                f.write(src.read())
            os.replace(f"{dst}.part", dst)
        return len(chunk)

    with ThreadPoolExecutor(8) as ex:
        print("FaceForensics++ videos:", sum(ex.map(job, [pick[i::8] for i in range(8)])), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--ff-per-method", type=int, default=170)
    ap.add_argument("--wdf-tars", type=int, default=12, help="tars per WildDeepfake train split (test uses 10)")
    a = ap.parse_args()
    a.data.mkdir(parents=True, exist_ok=True)
    get("thenewsupercell/celeb-df-image-dataset", parquet("thenewsupercell/celeb-df-image-dataset"), a.data)
    get("SpeechAntiSpoofingBenchmarks/InTheWild", parquet("SpeechAntiSpoofingBenchmarks/InTheWild"), a.data)
    get("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", ["data/test-00000-of-00009.parquet", "data/test-00005-of-00009.parquet"], a.data)
    get("SpeechAntiSpoofingBenchmarks/ASVspoof5", ["data/test-00000-of-00200.parquet", "data/test-00100-of-00200.parquet"], a.data)
    wd = list_repo_files("xingjunm/WildDeepfake", repo_type="dataset")
    pick: list[str] = []
    for split, n in (("real_train", a.wdf_tars), ("fake_train", a.wdf_tars), ("real_test", 10), ("fake_test", 10)):
        pick += sorted(f for f in wd if f"/{split}/" in f)[:n]
    get("xingjunm/WildDeepfake", pick, a.data)
    faceforensics(a.data, a.ff_per_method)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
