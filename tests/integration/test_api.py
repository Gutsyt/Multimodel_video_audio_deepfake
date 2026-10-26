"""API integration tests.

TEST DOUBLES: unless a test says otherwise, detectors are stubs from ``tests/stubs.py`` that return
fixed numbers (their model cards carry ``is_test_fixture=True``). These tests verify upload
validation, job lifecycle, status handling and cleanup. They say NOTHING about detection quality.
The one test marked "REAL WEIGHTS" runs the vendored pretrained AASIST through the API.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from avdd.api.app import create_app
from avdd.api.models import DetectorManager, bundle_from_detectors
from avdd.api.settings import Settings
from avdd.api.store import Store
from avdd.video.faces import HaarFaceDetector
from tests.conftest import requires_ffmpeg
from tests.integration.helpers import OCTET, Api, doubles
from tests.stubs import BlockingAudio, StubAudio, StubBoxes, StubFace

pytestmark = requires_ffmpeg
@pytest.fixture(scope="module")
def audio_only_mp4(tmp_path_factory: pytest.TempPathFactory, face_dir: Path) -> Path:
    out = tmp_path_factory.mktemp("audio_only") / "a.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(face_dir / "face_audio.mp4"), "-vn", "-c:a", "copy", str(out)],
                   check=True, timeout=60)  # fmt: skip
    return out


# ---------------------------------------------------------------- basics
def test_health_ready_capabilities(make_api: Callable[..., Api]) -> None:
    api = make_api(doubles())
    assert api.c.get("/api/health").json() == {"status": "ok"}
    assert api.c.get("/api/ready").status_code == 200
    cap = api.c.get("/api/capabilities").json()
    assert cap["models_state"] == "ready" and cap["stages"] == ["probing", "audio", "visual", "fusion"]
    assert {m["role"]: m["state"] for m in cap["models"]} == {"audio": "loaded", "visual": "loaded"}
    assert all(m["card"]["is_test_fixture"] for m in cap["models"])
    assert cap["limits"]["max_duration_s"] == 60.0 and "mov" in cap["limits"]["accepted_containers"]
    assert set(cap["limits"]["accepted"]) == {"video", "audio", "image"}
    assert ".wav" in cap["limits"]["accepted"]["audio"]["extensions"]
    assert ".png" in cap["limits"]["accepted"]["image"]["extensions"]
    assert "No training" in cap["evaluation_status"] and "deleted" in cap["retention"]


def test_cors_allows_the_vite_dev_origin(make_api: Callable[..., Api]) -> None:
    api = make_api(doubles())
    r = api.c.options("/api/analyses", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                                 "Access-Control-Request-Headers": "content-type"})  # fmt: skip
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://localhost:5173"


# ---------------------------------------------------------------- success with test doubles
def test_successful_analysis_lifecycle_with_doubles(make_api: Callable[..., Api], face_dir: Path) -> None:
    api = make_api(doubles())
    r = api.upload(face_dir / "face_audio.mp4")
    assert r.status_code == 202 and r.json()["status"] == "queued" and len(r.json()["id"]) == 32
    job = api.wait(r.json()["id"])
    assert job["status"] == "completed" and job["stage"] is None and job["error"] is None
    res = job["result"]
    assert res["is_test_fixture"] is True and res["outcome"] == "analyzed"
    assert res["modalities"]["audio"]["status"] == res["modalities"]["visual"]["status"] == "available"
    assert res["fusion"]["state"] == "multimodal" and res["fusion"]["fused_score"] == pytest.approx(0.5)
    assert res["fusion"]["calibrated"] is False and res["interpretation"]["validated_on_project_data"] is False
    assert res["modalities"]["audio"]["segments"][0]["start_s"] == 0.0
    assert all("t_s" in f for f in res["modalities"]["visual"]["frames"])
    text = json.dumps(job)
    assert str(api.settings.data_dir) not in text and "face_audio" not in text
    api.assert_clean()

    hist = api.c.get("/api/analyses").json()["items"]
    assert hist[0]["id"] == job["id"] and hist[0]["fusion_state"] == "multimodal" and hist[0]["is_test_fixture"] is True
    rep = api.c.get(f"/api/analyses/{job['id']}/report")
    assert rep.status_code == 200 and "attachment" in rep.headers["content-disposition"]
    assert rep.json()["result"]["fusion"]["state"] == "multimodal"
    assert api.c.delete(f"/api/analyses/{job['id']}").status_code == 204
    assert api.c.get(f"/api/analyses/{job['id']}").status_code == 404
    assert api.c.get("/api/analyses").json()["items"] == []


def test_progress_reports_real_pipeline_stage_while_running(make_api: Callable[..., Api], face_dir: Path) -> None:
    blocker = BlockingAudio(0.6)
    api = make_api(doubles(audio=blocker))
    job_id = api.upload(face_dir / "face_audio.mp4").json()["id"]
    try:
        assert blocker.entered.wait(15), "inference never started"
        mid = api.c.get(f"/api/analyses/{job_id}").json()
        assert mid["status"] == "processing" and mid["stage"] == "audio" and mid["result"] is None
        d = api.c.delete(f"/api/analyses/{job_id}")
        assert d.status_code == 409 and d.json()["error"]["code"] == "analysis_in_progress"
    finally:
        blocker.release.set()
    assert api.wait(job_id)["status"] == "completed"
    api.assert_clean()


# ---------------------------------------------------------------- missing / unreliable modalities
def test_video_without_audio_track(make_api: Callable[..., Api], face_dir: Path) -> None:
    res = make_api(doubles()).run(face_dir / "face_noaudio.mp4")["result"]
    a = res["modalities"]["audio"]
    assert a["status"] == "missing" and a["score"] is None
    assert res["fusion"]["state"] == "single_modality" and res["fusion"]["fused_score"] is None
    assert res["assessment"]["basis"] == "visual_only" and res["assessment"]["score"] == pytest.approx(0.2)
    assert res["fusion"]["excluded_modalities"] == {"audio": "missing"}


def test_audio_only_media_has_missing_visual(make_api: Callable[..., Api], audio_only_mp4: Path) -> None:
    res = make_api(doubles()).run(audio_only_mp4)["result"]
    assert res["modalities"]["visual"]["status"] == "missing" and res["modalities"]["visual"]["score"] is None
    assert res["fusion"]["state"] == "single_modality" and res["fusion"]["fused_score"] is None
    assert res["assessment"]["basis"] == "audio_only" and res["assessment"]["score"] == pytest.approx(0.8)


def test_silent_audio_is_unreliable_not_authentic(make_api: Callable[..., Api], face_dir: Path) -> None:
    res = make_api(doubles()).run(face_dir / "face_silent.mp4")["result"]
    assert res["modalities"]["audio"]["status"] == "unreliable" and res["modalities"]["audio"]["score"] is None
    assert res["fusion"]["excluded_modalities"] == {"audio": "unreliable"}


def test_no_face_makes_visual_unreliable_and_nothing_usable_is_insufficient(make_api: Callable[..., Api], media_dir: Path) -> None:
    # testsrc pattern has no face; with no audio track either, nothing usable remains.
    api = make_api(bundle_from_detectors(StubAudio(), StubFace(), HaarFaceDetector()))
    res = api.run(media_dir / "video_only.mp4")["result"]
    assert res["modalities"]["visual"]["status"] == "unreliable" and res["modalities"]["audio"]["status"] == "missing"
    assert res["fusion"]["state"] == "insufficient_evidence" and res["fusion"]["fused_score"] is None
    assert res["fusion"]["flagged_at_default_threshold"] is None


# ---------------------------------------------------------------- upload validation
@pytest.mark.parametrize("name", ["garbage.mp4", "truncated.mp4", "notes.txt"])
def test_corrupted_or_non_media_is_rejected_and_cleaned(make_api: Callable[..., Api], media_dir: Path, name: str) -> None:
    api = make_api(doubles())
    r = api.upload(media_dir / name)
    assert r.status_code == 422 and r.json()["error"]["code"] == "unreadable_media"
    assert str(media_dir) not in r.text
    assert api.c.get("/api/analyses").json()["items"] == []
    api.assert_clean()


def test_unsupported_but_readable_media_is_rejected_by_content_not_extension(
    make_api: Callable[..., Api], kinds_dir: Path
) -> None:
    api = make_api(doubles())
    r = api.upload(kinds_dir / "animated.gif")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_format"
    assert r.json()["error"]["details"]["detected_container"] == "gif"
    api.assert_clean()


def test_empty_upload(make_api: Callable[..., Api]) -> None:
    api = make_api(doubles())
    r = api.upload(b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_upload"
    api.assert_clean()


def test_oversized_upload_rejected_declared_and_streamed(make_api: Callable[..., Api]) -> None:
    api = make_api(doubles(), max_upload_bytes=1000)
    declared = api.upload(b"x" * 5000)
    assert declared.status_code == 413 and declared.json()["error"]["code"] == "file_too_large"
    streamed = api.c.post("/api/analyses", content=iter([b"x" * 400] * 10), headers=OCTET)  # no Content-Length
    assert streamed.status_code == 413 and streamed.json()["error"]["code"] == "file_too_large"
    api.assert_clean()


def test_duration_limit(make_api: Callable[..., Api], face_dir: Path) -> None:
    api = make_api(doubles(), max_duration_s=1.0)
    r = api.upload(face_dir / "face_audio.mp4")
    err = r.json()["error"]
    assert r.status_code == 422 and err["code"] == "duration_exceeded" and err["details"]["max_duration_s"] == 1.0
    api.assert_clean()


# ---------------------------------------------------------------- model availability / failures
def test_partial_model_failure_is_visible_and_not_authentic(make_api: Callable[..., Api], face_dir: Path) -> None:
    bundle = bundle_from_detectors(None, StubFace(0.3), StubBoxes(), failures={"audio": "weights_incompatible"})
    api = make_api(bundle)
    cap = api.c.get("/api/capabilities").json()
    audio = next(m for m in cap["models"] if m["role"] == "audio")
    assert audio["state"] == "failed" and audio["error_code"] == "weights_incompatible"
    res = api.run(face_dir / "face_audio.mp4")["result"]
    assert res["modalities"]["audio"]["status"] == "not_analyzed" and res["modalities"]["audio"]["score"] is None
    assert res["fusion"]["state"] == "single_modality" and res["fusion"]["excluded_modalities"] == {"audio": "not_analyzed"}


def test_real_loader_failure_reports_unavailable(make_api: Callable[..., Api], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, face_dir: Path) -> None:
    monkeypatch.setenv("AVDD_THIRD_PARTY_DIR", str(tmp_path / "does_not_exist"))
    api = make_api(None, audio_model="AASIST", video_model="none")  # real loader, vendored files unreachable
    end = time.time() + 20
    while api.c.get("/api/capabilities").json()["models_state"] == "loading" and time.time() < end:
        time.sleep(0.05)
    cap = api.c.get("/api/capabilities").json()
    assert cap["models_state"] == "unavailable"
    assert next(m for m in cap["models"] if m["role"] == "audio")["error_code"] in ("files_missing", "torch_missing")
    assert api.c.get("/api/ready").status_code == 503
    r = api.upload(face_dir / "face_audio.mp4")
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_models_available"
    api.assert_clean()


def test_models_still_loading_returns_503(make_api: Callable[..., Api], face_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DetectorManager, "start", lambda self: None)
    api = make_api(None)
    assert api.c.get("/api/capabilities").json()["models_state"] == "loading"
    r = api.upload(face_dir / "face_audio.mp4")
    assert r.status_code == 503 and r.json()["error"]["code"] == "models_loading"
    api.assert_clean()


def test_model_inference_error_degrades_to_not_analyzed(make_api: Callable[..., Api], face_dir: Path) -> None:
    res = make_api(doubles(audio=StubAudio(mode="raise"))).run(face_dir / "face_audio.mp4")["result"]
    assert res["modalities"]["audio"]["status"] == "not_analyzed" and res["modalities"]["audio"]["score"] is None
    assert "boom" not in json.dumps(res) and "secret" not in json.dumps(res)


def test_unexpected_crash_fails_job_without_leaking_and_cleans_up(make_api: Callable[..., Api], face_dir: Path) -> None:
    class ExplodingBoxes(StubBoxes):
        def detect(self, image_bgr: Any) -> Any:
            raise RuntimeError("secret internal failure at /private/path")

    api = make_api(bundle_from_detectors(StubAudio(), StubFace(), ExplodingBoxes()))
    job = api.run(face_dir / "face_audio.mp4")
    assert job["status"] == "failed" and job["result"] is None
    assert job["error"]["code"] == "internal_error"
    assert "secret" not in json.dumps(job) and "/private" not in json.dumps(job)
    api.assert_clean()
    assert api.c.get("/api/analyses").json()["items"][0]["error_code"] == "internal_error"


# ---------------------------------------------------------------- queueing, restarts, errors
def test_queue_full_returns_429_then_recovers(make_api: Callable[..., Api], face_dir: Path) -> None:
    blocker = BlockingAudio()
    api = make_api(doubles(audio=blocker), max_pending=1)
    first = api.upload(face_dir / "face_audio.mp4")
    assert first.status_code == 202
    try:
        second = api.upload(face_dir / "face_audio.mp4")
        assert second.status_code == 429 and second.json()["error"]["code"] == "queue_full"
    finally:
        blocker.release.set()
    api.wait(first.json()["id"])
    third = api.upload(face_dir / "face_noaudio.mp4")
    assert third.status_code == 202  # the slot was released after the first job finished
    api.wait(third.json()["id"])
    api.assert_clean()


def test_failed_file_deletion_does_not_wedge_the_queue(make_api: Callable[..., Api], face_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a Windows file lock: deleting the temp upload raises, the slot must still be freed."""
    real_unlink = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".upload":
            raise PermissionError("file is locked")
        real_unlink(self, missing_ok=missing_ok)

    api = make_api(doubles(), max_pending=1)
    monkeypatch.setattr(Path, "unlink", locked)
    first = api.run(face_dir / "face_audio.mp4")
    assert first["status"] == "completed"  # the analysis result is unaffected
    second = api.upload(face_dir / "face_noaudio.mp4")  # would be 429 if the slot had leaked
    assert second.status_code == 202
    api.wait(second.json()["id"])
    monkeypatch.undo()
    for leftover in api.settings.upload_dir.iterdir():  # what the next-start sweep would remove
        leftover.unlink()


def test_restart_marks_interrupted_jobs_failed_and_removes_leftover_uploads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    store = Store(data / "analyses.sqlite3")
    store.create("a" * 32)
    store.set_stage("a" * 32, "processing", "audio")
    (data / "uploads_tmp").mkdir(parents=True)
    (data / "uploads_tmp" / "leftover.upload").write_bytes(b"x")
    with TestClient(create_app(Settings(data_dir=data), doubles())) as c:
        body = c.get(f"/api/analyses/{'a' * 32}").json()
        assert body["status"] == "failed" and body["error"]["code"] == "interrupted"
        assert list((data / "uploads_tmp").iterdir()) == []


def test_error_shape_for_unknown_ids_and_routes(make_api: Callable[..., Api]) -> None:
    api = make_api(doubles())
    for url in ("/api/analyses/" + "b" * 32, "/api/analyses/not-an-id", "/api/nothing"):
        r = api.c.get(url)
        assert r.status_code == 404 and set(r.json()["error"]) >= {"code", "message"}
    assert api.c.get("/api/analyses/" + "b" * 32 + "/report").status_code == 404


# ---------------------------------------------------------------- REAL WEIGHTS
def test_real_weights_through_the_api_not_test_doubles(make_api: Callable[..., Api], face_dir: Path) -> None:
    """REAL WEIGHTS: vendored pretrained AASIST-L inference through the HTTP API.

    Input is a synthetic sine tone, so the score itself is meaningless; this checks that real
    model inference is wired end to end and is not flagged as a test fixture.
    """
    pytest.importorskip("torch")
    from avdd.audio.aasist_detector import AasistDetector

    det = AasistDetector("AASIST-L")
    det.load()
    api = make_api(bundle_from_detectors(det, None, None))
    res = api.run(face_dir / "face_audio.mp4")["result"]
    assert res["is_test_fixture"] is False and res["models"][0]["name"] == "AASIST-L"
    assert res["modalities"]["audio"]["status"] == "available" and 0.0 <= res["modalities"]["audio"]["score"] <= 1.0
    assert res["modalities"]["visual"]["status"] == "not_analyzed" and res["fusion"]["state"] == "single_modality"
    api.assert_clean()
