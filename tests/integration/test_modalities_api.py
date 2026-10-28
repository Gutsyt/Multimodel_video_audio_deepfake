"""Optional-modality behaviour through the HTTP API.

TEST DOUBLES: detectors are stubs from ``tests/stubs.py`` (fixed scores, ``is_test_fixture=True``);
media are synthetic (sine tones, a public-domain photo, ffmpeg patterns). These tests verify which
branches run, how missing evidence is reported and that nothing is ever substituted for it. They
say NOTHING about detection quality; see ``tests/real`` for the real-model smoke tests.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("skimage")

from avdd.api.models import bundle_from_detectors
from avdd.video.faces import YuNetFaceDetector
from tests.conftest import requires_ffmpeg
from tests.integration.helpers import Api, doubles
from tests.stubs import StubAudio, StubFace

pytestmark = requires_ffmpeg

HAAR = bundle_from_detectors(StubAudio(0.8), StubFace(0.2), YuNetFaceDetector())


def result(api: Api, path: Path) -> dict:  # type: ignore[type-arg]
    job = api.run(path)
    assert job["status"] == "completed", job
    return job["result"]  # type: ignore[no-any-return]


# ------------------------------------------------------------ audio-only uploads
@pytest.mark.parametrize("name", ["tone.mp3", "tone.flac", "tone.ogg", "tone.m4a"])
def test_audio_only_upload_runs_the_audio_branch_alone(make_api: Callable[..., Api], kinds_dir: Path, name: str) -> None:
    api = make_api(doubles())
    res = result(api, kinds_dir / name)
    assert res["media"]["kind"] == "audio" and res["media"]["has_video"] is False
    assert res["modalities"]["audio"]["status"] == "available"
    assert res["modalities"]["audio"]["score"] == pytest.approx(0.8)
    v = res["modalities"]["visual"]
    assert v["status"] == "missing" and v["score"] is None and v["reasons"] == ["this audio file has no video stream"]
    a = res["assessment"]
    assert (a["verdict"], a["basis"], a["score"]) == ("likely_manipulated", "audio_only", pytest.approx(0.8))
    assert res["modalities"]["audio"]["verdict"] == "likely_manipulated" and v["verdict"] is None
    api.assert_clean()


def test_wav_is_accepted_as_audio(make_api: Callable[..., Api], media_dir: Path) -> None:
    res = result(make_api(doubles()), media_dir / "audio_only.wav")
    assert res["media"]["kind"] == "audio" and res["assessment"]["basis"] == "audio_only"


# ------------------------------------------------------------ image-only uploads
@pytest.mark.parametrize("name", ["face.jpg", "face.png", "face.webp", "face.bmp"])
def test_image_only_upload_runs_the_visual_branch_alone(make_api: Callable[..., Api], kinds_dir: Path, name: str) -> None:
    api = make_api(bundle_from_detectors(StubAudio(0.9), StubFace(0.2), YuNetFaceDetector()))
    res = result(api, kinds_dir / name)
    assert res["media"]["kind"] == "image"
    a = res["modalities"]["audio"]
    assert a["status"] == "missing" and a["score"] is None and a["reasons"] == ["this image has no audio"]
    v = res["modalities"]["visual"]
    assert v["status"] == "available" and v["source"] == "image" and v["score"] == pytest.approx(0.2)
    assert v["n_frames_sampled"] == 1 and v["frames"][0]["t_s"] is None
    assert res["assessment"]["verdict"] == "likely_authentic" and res["assessment"]["basis"] == "visual_only"
    api.assert_clean()


def test_image_without_a_detectable_face_is_insufficient_evidence(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    api = make_api(HAAR)
    res = result(api, kinds_dir / "noface.jpg")
    v = res["modalities"]["visual"]
    assert v["status"] == "unreliable" and v["score"] is None and "no face" in v["reasons"][0]
    assert res["assessment"]["verdict"] == "insufficient_evidence" and res["assessment"]["score"] is None
    api.assert_clean()


def test_image_pixel_cap_is_enforced_before_analysis(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    api = make_api(doubles(), max_image_pixels=1000)
    r = api.upload(kinds_dir / "face.png")
    assert r.status_code == 422 and r.json()["error"]["code"] == "image_too_large"
    assert api.c.get("/api/analyses").json()["items"] == []
    api.assert_clean()


def test_tiff_image_is_unsupported(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    api = make_api(doubles())
    r = api.upload(kinds_dir / "face.tiff")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_format"
    api.assert_clean()


# ------------------------------------------------------------ video: each modality combination
def test_video_with_audio_but_no_face_runs_audio_only(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    api = make_api(HAAR)
    res = result(api, kinds_dir / "noface_audio.mp4")
    assert res["media"]["kind"] == "video"
    v = res["modalities"]["visual"]
    assert v["status"] == "unreliable" and v["score"] is None and "need at least" in v["reasons"][0]
    assert res["modalities"]["audio"]["status"] == "available"
    assert res["assessment"]["basis"] == "audio_only" and res["assessment"]["verdict"] == "likely_manipulated"
    api.assert_clean()


def test_video_with_face_but_no_audio_runs_visual_only(make_api: Callable[..., Api], face_dir: Path) -> None:
    res = result(make_api(doubles()), face_dir / "face_noaudio.mp4")
    assert res["modalities"]["audio"]["status"] == "missing"
    assert res["modalities"]["audio"]["reasons"] == ["no audio track in this file"]
    assert res["assessment"]["basis"] == "visual_only"
    assert res["assessment"]["score"] == pytest.approx(0.2)


def test_video_with_silent_audio_track_runs_visual_only(make_api: Callable[..., Api], face_dir: Path) -> None:
    res = result(make_api(doubles()), face_dir / "face_silent.mp4")
    a = res["modalities"]["audio"]
    assert a["status"] == "unreliable" and a["score"] is None and a["verdict"] is None
    assert res["assessment"]["basis"] == "visual_only"


def test_video_with_audio_and_face_uses_both_branches(make_api: Callable[..., Api], face_dir: Path) -> None:
    a = result(make_api(doubles()), face_dir / "face_audio.mp4")["assessment"]
    assert (a["verdict"], a["basis"]) == ("likely_manipulated", "multimodal")  # one manipulated branch is enough
    assert a["branch_verdicts"] == {"audio": "likely_manipulated", "visual": "likely_authentic"}


def test_video_with_neither_usable_audio_nor_face_is_insufficient_evidence(make_api: Callable[..., Api], media_dir: Path) -> None:
    api = make_api(HAAR)
    res = result(api, media_dir / "video_only.mp4")
    assert res["assessment"]["verdict"] == "insufficient_evidence" and res["assessment"]["basis"] == "none"
    assert all(m["score"] is None and m["verdict"] is None for m in res["modalities"].values())
    assert "says nothing about authenticity" in res["assessment"]["summary"]


# ------------------------------------------------------------ validation errors
@pytest.mark.parametrize("name,status,code", [
    ("empty.wav", 400, "empty_upload"),
    ("fake.jpg", 422, "unreadable_media"),
    ("truncated.jpg", 422, "unreadable_media"),
    ("truncated.mp3", 422, "unreadable_media"),
    ("animated.gif", 415, "unsupported_format"),
])  # fmt: skip
def test_empty_corrupted_and_unsupported_files_get_clear_errors(
    make_api: Callable[..., Api], kinds_dir: Path, name: str, status: int, code: str
) -> None:
    api = make_api(doubles())
    r = api.upload(kinds_dir / name)
    assert r.status_code == status and r.json()["error"]["code"] == code, r.text
    assert str(kinds_dir) not in r.text
    assert api.c.get("/api/analyses").json()["items"] == []
    api.assert_clean()


def test_unsupported_message_names_all_three_kinds(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    msg = make_api(doubles()).upload(kinds_dir / "animated.gif").json()["error"]["message"]
    assert "video" in msg and "audio" in msg and "image" in msg


# ------------------------------------------------------------ history, clear, schema tolerance
def test_history_summary_reports_kind_and_decisions(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    api = make_api(doubles())
    result(api, kinds_dir / "tone.flac")
    item = api.c.get("/api/analyses").json()["items"][0]
    assert item["media_kind"] == "audio" and item["verdict"] == "likely_manipulated" and item["basis"] == "audio_only"
    assert item["score"] == pytest.approx(0.8) and item["audio_score"] == pytest.approx(0.8)


def test_clear_history_deletes_finished_records_only(make_api: Callable[..., Api], kinds_dir: Path) -> None:
    api = make_api(doubles())
    result(api, kinds_dir / "tone.flac")
    result(api, kinds_dir / "face.png")
    r = api.c.delete("/api/analyses")
    assert r.status_code == 200 and r.json() == {"deleted": 2}
    assert api.c.get("/api/analyses").json()["items"] == []


def test_results_from_the_old_schema_open_as_legacy(make_api: Callable[..., Api]) -> None:
    api = make_api(doubles())
    legacy = {
        "schema_version": "0.1", "avdd_version": "0.0.1", "created_utc": "2026-10-20T10:00:00+00:00",
        "outcome": "analyzed",
        "media": {"container": "mov,mp4", "duration_s": 3.0, "has_video": True, "has_audio": True},
        "modalities": {"audio": {"status": "available", "score": 0.7, "reasons": []},
                       "visual": {"status": "missing", "score": None, "reasons": []}},
        "fusion": {"method": "m", "state": "single_modality", "fused_score": 0.7,
                   "flagged_at_default_threshold": True, "threshold": 0.5, "weights_used": {"audio": 1.0},
                   "branch_flags_at_default_threshold": {"audio": True}, "excluded_modalities": {"visual": "missing"},
                   "weights_validated": False, "threshold_validated": False, "calibrated": False, "notes": []},
        "models": [], "is_test_fixture": False,
        "interpretation": {"score_meaning": "s", "calibrated": False, "validated_on_project_data": False,
                           "flag_meaning": "f", "not_forensic_proof": "n"},
        "analyzed_limit_s": 60.0, "total_runtime_s": 1.0,
    }  # fmt: skip
    job_id = "a" * 32
    api.c.app.state.store.create(job_id)  # type: ignore[attr-defined]
    with closing(sqlite3.connect(api.settings.db_path)) as c, c:
        c.execute("UPDATE analyses SET status='completed', result_json=?, summary_json=? WHERE id=?",
                  (json.dumps(legacy), json.dumps({"fusion_state": "single_modality", "fused_score": 0.7}), job_id))
    got = api.c.get(f"/api/analyses/{job_id}").json()
    assert got["status"] == "completed" and got["result"] is None and got["error"]["code"] == "legacy_result"
    item = api.c.get("/api/analyses").json()["items"][0]
    assert item["id"] == job_id and item["verdict"] is None
