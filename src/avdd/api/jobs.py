"""Single-worker job runner: one analysis at a time, bounded queue, guaranteed cleanup."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from avdd.api.models import DetectorManager
from avdd.api.schemas import AnalysisResult
from avdd.api.settings import Settings
from avdd.api.store import Store
from avdd.common.errors import AvddError
from avdd.pipeline.analyze import AnalysisConfig, analyze_media
from avdd.video.branch import VideoBranchConfig

logger = logging.getLogger(__name__)


def summarize(result: AnalysisResult) -> dict[str, object]:
    a, v = result.modalities.get("audio"), result.modalities.get("visual")
    asm = result.assessment
    return {
        "duration_s": result.media.duration_s,
        "media_kind": result.media.kind,
        "assessment_status": asm.status if asm else None,
        "basis": asm.basis if asm else None,
        "audio_decision": a.decision if a else None,
        "visual_decision": v.decision if v else None,
        "fusion_state": result.fusion.state,
        "fused_score": result.fusion.fused_score,
        "flagged": result.fusion.flagged_at_default_threshold,
        "audio_status": a.status if a else None,
        "audio_score": a.score if a else None,
        "visual_status": v.status if v else None,
        "visual_score": v.score if v else None,
        "is_test_fixture": result.is_test_fixture,
    }


class JobRunner:
    def __init__(self, settings: Settings, store: Store, manager: DetectorManager) -> None:
        self._s, self._store, self._manager = settings, store, manager
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avdd-analysis")
        self._lock = threading.Lock()
        self._pending = 0

    def reserve(self) -> bool:
        """Reserve a queue slot; False when the queue is full."""
        with self._lock:
            if self._pending >= self._s.max_pending:
                return False
            self._pending += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    def submit(self, job_id: str, path: Path) -> None:
        """Queue a job. The runner now owns ``path`` and the reserved slot."""
        try:
            self._executor.submit(self._run, job_id, path)
        except BaseException:
            self._discard(path)
            self.release()
            raise

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, path: Path) -> None:
        try:
            self._store.set_stage(job_id, "processing", "probing")
            bundle = self._manager.bundle()
            if bundle is None:
                raise AvddError("models_unavailable", "The analysis models are not available.")
            s = self._s
            cfg = AnalysisConfig(max_seconds=s.max_duration_s,
                                 video=VideoBranchConfig(n_frames=s.n_frames, max_seconds=s.max_duration_s))
            raw = analyze_media(path, bundle.audio, bundle.video, bundle.face, cfg,
                                on_stage=lambda st: self._store.set_stage(job_id, "processing", st))
            result = AnalysisResult.model_validate(raw)
            self._store.complete(job_id, result.model_dump_json(), json.dumps(summarize(result)))
            self._store.prune(s.history_max_items)
        except AvddError as exc:
            self._store.fail(job_id, json.dumps({"code": exc.code, "message": exc.message}))
        except Exception:
            logger.exception("analysis job failed")
            self._store.fail(job_id, json.dumps({"code": "internal_error",
                                                 "message": "The analysis failed unexpectedly."}))
        finally:
            self._discard(path)
            self.release()  # always runs, even if the file could not be deleted

    @staticmethod
    def _discard(path: Path) -> None:
        """Delete the temporary upload. A failure (e.g. a Windows file lock) must never stop the
        queue slot from being released; leftovers are removed by the sweep at the next start."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not delete a temporary upload; it will be removed at next start")
