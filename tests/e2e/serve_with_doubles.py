#!/usr/bin/env python3
"""Start the API with TEST DOUBLE detectors (fixed numbers, flagged is_test_fixture=True).

For wiring checks of the UI/API only (e.g. the frontend e2e test). Not a detector.
Run from the repo root:  PYTHONPATH=src:. python tests/e2e/serve_with_doubles.py --port 8078
"""

from __future__ import annotations

import argparse

import uvicorn

from avdd.api.app import create_app
from avdd.api.models import bundle_from_detectors
from avdd.api.settings import Settings
from avdd.video.faces import YuNetFaceDetector
from tests.stubs import StubAudio, StubFace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8078)
    args = ap.parse_args()
    bundle = bundle_from_detectors(StubAudio(0.8), StubFace(0.2), YuNetFaceDetector())
    uvicorn.run(create_app(Settings.from_env(), bundle), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
