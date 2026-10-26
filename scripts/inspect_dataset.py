#!/usr/bin/env python3
"""Generate a verified inventory of a dataset directory.

Run from the repository root:

    python scripts/inspect_dataset.py --root /path/to/FakeAVCeleb --out inventory_out

Outputs (in --out): inventory.json, INVENTORY.md, media_probe.jsonl.
The tool records observed facts only; it never infers labels or splits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from avdd.data.inventory import InspectionConfig, run_inspection


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="dataset root directory")
    ap.add_argument("--out", type=Path, default=Path("inventory_out"), help="output directory")
    ap.add_argument("--no-probe", action="store_true", help="skip ffprobe on media files")
    ap.add_argument("--max-probe", type=int, default=None,
                    help="probe only a deterministic random sample of N media files")
    ap.add_argument("--seed", type=int, default=0, help="seed for --max-probe sampling")
    ap.add_argument("--hash", action="store_true", help="sha256 probed media files and report exact duplicates")
    ap.add_argument("--workers", type=int, default=4, help="parallel ffprobe workers")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-file ffprobe timeout (s)")
    ap.add_argument("--dir-depth", type=int, default=3, help="directory aggregation depth")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = InspectionConfig(
        root=args.root, out_dir=args.out, probe=not args.no_probe, max_probe=args.max_probe,
        seed=args.seed, hash_media=args.hash, workers=args.workers,
        probe_timeout_s=args.timeout, dir_depth=args.dir_depth,
    )
    try:
        inv = run_inspection(cfg)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"files={inv['totals']['n_files']} media={inv['totals']['n_media_files']} -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
