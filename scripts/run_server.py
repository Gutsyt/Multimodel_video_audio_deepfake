#!/usr/bin/env python3
"""Start the local API server (and the built frontend if present).

    python scripts/run_server.py                  # http://127.0.0.1:8000
    python scripts/run_server.py --port 8001

Configuration is read from AVDD_* environment variables (see docs/runbook_local.md).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from avdd.api.app import create_app
from avdd.api.settings import Settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if settings.frontend_dist is None and "AVDD_FRONTEND_DIST" not in os.environ:
        built = Path("frontend/dist")
        if (built / "index.html").is_file():
            settings = Settings(**{**settings.__dict__, "frontend_dist": built})
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
