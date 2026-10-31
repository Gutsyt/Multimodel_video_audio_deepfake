"""Shared helpers for the API integration tests (an API wrapper and detector-double bundles)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from avdd.api.models import DetectorBundle, bundle_from_detectors
from avdd.api.settings import Settings
from tests.stubs import StubAudio, StubBoxes, StubFace

OCTET = {"Content-Type": "application/octet-stream"}


class Api:
    def __init__(self, client: TestClient, settings: Settings) -> None:
        self.c, self.settings = client, settings

    def upload(self, path: Path | bytes, **kw: Any) -> Any:
        data = path.read_bytes() if isinstance(path, Path) else path
        return self.c.post("/api/analyses", content=data, headers=OCTET, **kw)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
        end = time.time() + timeout
        while time.time() < end:
            body = self.c.get(f"/api/analyses/{job_id}").json()
            if body["status"] in ("completed", "failed"):
                return body  # type: ignore[no-any-return]
            time.sleep(0.05)
        raise AssertionError("analysis did not finish in time")

    def run(self, path: Path, timeout: float = 30.0) -> dict[str, Any]:
        r = self.upload(path)
        assert r.status_code == 202, r.text
        return self.wait(r.json()["id"], timeout)

    def assert_clean(self) -> None:
        assert list(self.settings.upload_dir.iterdir()) == [], "temporary upload files were left behind"


def doubles(audio: Any = None, video: Any = None, **kw: Any) -> DetectorBundle:
    audio = StubAudio(0.8) if audio is None else audio
    video = StubFace(0.2) if video is None else video
    return bundle_from_detectors(audio, video, StubBoxes(), **kw)
