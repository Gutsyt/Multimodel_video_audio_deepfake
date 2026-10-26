"""FastAPI application factory around the existing ``analyze_media`` pipeline."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from avdd import __version__
from avdd.api.errors import ApiError, install_error_handlers
from avdd.api.jobs import JobRunner
from avdd.api.models import DetectorBundle, DetectorManager
from avdd.api.schemas import (
    AnalysisCreated,
    AnalysisResult,
    AnalysisStatusOut,
    AnalysisSummary,
    ApiErrorBody,
    CapabilitiesOut,
    ErrorResponse,
    HistoryOut,
    LimitsOut,
    ModelStatusOut,
)
from avdd.api.settings import Settings
from avdd.api.store import Store
from avdd.api.uploads import receive_upload, validate_media
from avdd.pipeline.analyze import STAGES

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
RETENTION = (
    "Uploaded videos are processed from a temporary file that is deleted as soon as the analysis "
    "finishes or fails. Only the analysis result (scores, timestamps, media metadata; no file name "
    "and no media) is kept in a local database until you delete it."
)
EVALUATION = (
    "Pretrained-model inference only. No training was performed and no performance evaluation has "
    "been completed; scores are uncalibrated and unvalidated."
)


def _check_id(analysis_id: str) -> str:
    if not _ID_RE.match(analysis_id):
        raise ApiError(404, "not_found", "Analysis not found.")
    return analysis_id


def create_app(settings: Settings | None = None, detectors: DetectorBundle | None = None) -> FastAPI:
    """Build the app. ``detectors`` injects a prepared bundle (used by tests); otherwise real
    pretrained models are loaded in the background at startup."""
    s = settings or Settings.from_env()
    store = Store(s.db_path)
    manager = DetectorManager(s, detectors)
    runner = JobRunner(s, store, manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        s.upload_dir.mkdir(parents=True, exist_ok=True)
        for leftover in s.upload_dir.iterdir():  # raw uploads never survive a restart
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        store.mark_interrupted()
        manager.start()
        yield
        runner.shutdown()

    app = FastAPI(
        title="avdd: audio-visual deepfake analysis (research prototype)",
        version=__version__,
        description=EVALUATION,
        lifespan=lifespan,
        docs_url="/docs" if s.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if s.enable_docs else None,
    )
    install_error_handlers(app)
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins),
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def _headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    err = {"model": ErrorResponse}

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/ready", responses={503: err})
    async def ready() -> dict[str, str]:
        if manager.state != "ready":
            raise ApiError(503, "models_not_ready", f"Models are {manager.state}.")
        return {"status": "ready"}

    @app.get("/api/capabilities", response_model=CapabilitiesOut)
    async def capabilities() -> CapabilitiesOut:
        return CapabilitiesOut(
            avdd_version=__version__,
            models_state=manager.state,
            models=[ModelStatusOut.model_validate(m, from_attributes=True) for m in manager.statuses()],
            limits=LimitsOut(max_upload_mb=round(s.max_upload_bytes / 1e6, 1), max_duration_s=s.max_duration_s,
                             accepted_containers=sorted(s.allowed_containers), frames_sampled=s.n_frames,
                             max_pending_jobs=s.max_pending),
            retention=RETENTION,
            evaluation_status=EVALUATION,
            stages=list(STAGES),
        )

    @app.post("/api/analyses", status_code=202, response_model=AnalysisCreated,
              responses={400: err, 413: err, 415: err, 422: err, 429: err, 503: err})
    async def create_analysis(request: Request) -> AnalysisCreated:
        """Upload a video as the raw request body (``Content-Type: application/octet-stream``)."""
        if manager.state == "loading":
            raise ApiError(503, "models_loading", "The models are still loading; try again shortly.")
        if manager.state == "unavailable":
            raise ApiError(503, "no_models_available", "No analysis model could be loaded; see /api/capabilities.")
        if not runner.reserve():
            raise ApiError(429, "queue_full", "Too many analyses are pending; try again shortly.")
        path = None
        try:
            path = await receive_upload(request, s.upload_dir, s.max_upload_bytes)
            await run_in_threadpool(validate_media, path, s)
            job_id = uuid.uuid4().hex
            store.create(job_id)
            runner.submit(job_id, path)  # runner now owns the file and the slot
        except BaseException:
            if path is not None:
                path.unlink(missing_ok=True)
            runner.release()
            raise
        return AnalysisCreated(id=job_id, status="queued")

    def _status(row: dict[str, Any]) -> AnalysisStatusOut:
        result = AnalysisResult.model_validate_json(row["result_json"]) if row["result_json"] else None
        error = ApiErrorBody.model_validate_json(row["error_json"]) if row["error_json"] else None
        return AnalysisStatusOut(id=row["id"], status=row["status"], stage=row["stage"], stages=list(STAGES),
                                 created_utc=row["created_utc"], updated_utc=row["updated_utc"],
                                 result=result, error=error)

    @app.get("/api/analyses", response_model=HistoryOut)
    async def history() -> HistoryOut:
        items = []
        for r in store.list():
            summary = json.loads(r["summary_json"]) if r["summary_json"] else {}
            code = json.loads(r["error_json"])["code"] if r["error_json"] else None
            items.append(AnalysisSummary(id=r["id"], status=r["status"], created_utc=r["created_utc"],
                                         error_code=code, **summary))
        return HistoryOut(items=items)

    @app.get("/api/analyses/{analysis_id}", response_model=AnalysisStatusOut, responses={404: err})
    async def get_analysis(analysis_id: str) -> AnalysisStatusOut:
        row = store.get(_check_id(analysis_id))
        if row is None:
            raise ApiError(404, "not_found", "Analysis not found.")
        return _status(row)

    @app.get("/api/analyses/{analysis_id}/report", responses={404: err, 409: err})
    async def report(analysis_id: str) -> Response:
        """Download the stored result as a JSON report (no media, no file name)."""
        row = store.get(_check_id(analysis_id))
        if row is None:
            raise ApiError(404, "not_found", "Analysis not found.")
        if row["status"] != "completed" or not row["result_json"]:
            raise ApiError(409, "not_completed", "The analysis has not completed.")
        doc = {"report_type": "avdd-analysis-report", "analysis_id": analysis_id, "evaluation_status": EVALUATION,
               "result": json.loads(row["result_json"])}
        return Response(json.dumps(doc, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="avdd-analysis-{analysis_id[:8]}.json"'})

    @app.delete("/api/analyses/{analysis_id}", status_code=204, responses={404: err, 409: err})
    async def delete_analysis(analysis_id: str) -> Response:
        row = store.get(_check_id(analysis_id))
        if row is None:
            raise ApiError(404, "not_found", "Analysis not found.")
        if row["status"] in ("queued", "processing"):
            raise ApiError(409, "analysis_in_progress", "The analysis is still running; delete it when it finishes.")
        store.delete(analysis_id)
        return Response(status_code=204)

    if s.frontend_dist and (s.frontend_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=s.frontend_dist, html=True), name="frontend")

    app.state.settings, app.state.store, app.state.manager, app.state.runner = s, store, manager, runner
    return app


def purge_dir(path: Any) -> None:
    """Remove a directory tree (used by tests/cleanup helpers)."""
    shutil.rmtree(path, ignore_errors=True)
