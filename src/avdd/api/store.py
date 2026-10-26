"""SQLite persistence for analysis jobs/results. Stores NO raw media and NO file names."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT,
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL,
    result_json TEXT,
    summary_json TEXT,
    error_json TEXT
)
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, job_id: str) -> None:
        t = now_utc()
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO analyses (id, status, stage, created_utc, updated_utc) VALUES (?,?,?,?,?)",
                      (job_id, "queued", None, t, t))

    def set_stage(self, job_id: str, status: str, stage: str | None) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE analyses SET status=?, stage=?, updated_utc=? WHERE id=?",
                      (status, stage, now_utc(), job_id))

    def complete(self, job_id: str, result_json: str, summary_json: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE analyses SET status='completed', stage=NULL, result_json=?, summary_json=?, "
                      "updated_utc=? WHERE id=?", (result_json, summary_json, now_utc(), job_id))

    def fail(self, job_id: str, error_json: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE analyses SET status='failed', stage=NULL, error_json=?, updated_utc=? WHERE id=?",
                      (error_json, now_utc(), job_id))

    def get(self, job_id: str) -> dict[str, Any] | None:
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM analyses WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT id,status,created_utc,summary_json,error_json FROM analyses "
                             "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete(self, job_id: str) -> bool:
        with closing(self._conn()) as c, c:
            return c.execute("DELETE FROM analyses WHERE id=?", (job_id,)).rowcount > 0

    def delete_finished(self) -> int:
        """Delete every completed or failed record; running jobs are left alone."""
        with closing(self._conn()) as c, c:
            return c.execute("DELETE FROM analyses WHERE status IN ('completed','failed')").rowcount

    def prune(self, keep: int) -> None:
        with closing(self._conn()) as c, c:
            c.execute("DELETE FROM analyses WHERE status IN ('completed','failed') AND id NOT IN "
                      "(SELECT id FROM analyses ORDER BY rowid DESC LIMIT ?)", (keep,))

    def mark_interrupted(self) -> int:
        """Jobs left queued/processing by a previous server run can never finish: mark them failed."""
        err = '{"code": "interrupted", "message": "The server restarted before this analysis finished."}'
        with closing(self._conn()) as c, c:
            return c.execute("UPDATE analyses SET status='failed', stage=NULL, error_json=?, updated_utc=? "
                             "WHERE status IN ('queued','processing')", (err, now_utc())).rowcount
