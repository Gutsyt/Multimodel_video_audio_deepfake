"""Fixtures for the API integration tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from avdd.api.app import create_app
from avdd.api.models import DetectorBundle
from avdd.api.settings import Settings
from tests.integration.helpers import Api


@pytest.fixture()
def make_api(tmp_path: Path) -> Iterator[Callable[..., Api]]:
    clients: list[TestClient] = []

    def make(detectors: DetectorBundle | None = None, data: str = "data", **overrides: Any) -> Api:
        settings = Settings(data_dir=tmp_path / data, **overrides)
        client = TestClient(create_app(settings, detectors))
        client.__enter__()
        clients.append(client)
        return Api(client, settings)

    yield make
    for c in clients:
        c.__exit__(None, None, None)
