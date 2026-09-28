"""Outbound worker job queue (Phase 3.2 control plane on the ingest host).

Workers poll ``POST /v1/workers/claim``; laptops (or CI) enqueue with
``POST /v1/jobs``. Jobs are JSON files under ``<data>/jobs/``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class JobQueue:
    def __init__(self, root: Path, *, clock=time.time) -> None:
        self.root = Path(root).resolve() / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def enqueue(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("id") or f"job_{uuid.uuid4().hex[:16]}")
        now = self._clock()
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "payload": payload.get("payload") if "payload" in payload else payload,
            "worker_id": None,
            "result": None,
        }
        with self._lock:
            self._path(job_id).write_text(json.dumps(job, indent=2), encoding="utf-8")
        return job

    def claim(self, *, worker_id: str, gpu: dict[str, Any] | None = None) -> dict[str, Any] | None:
        del gpu  # inventory is recorded on the worker heartbeat side for now
        with self._lock:
            for path in sorted(self.root.glob("*.json")):
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(job, dict) or job.get("status") != "queued":
                    continue
                job["status"] = "running"
                job["worker_id"] = worker_id
                job["updated_at"] = self._clock()
                path.write_text(json.dumps(job, indent=2), encoding="utf-8")
                return job
        return None

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            path = self._path(job_id)
            if not path.is_file():
                return None
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if status is not None:
                job["status"] = status
            if result is not None:
                job["result"] = result
            job["updated_at"] = self._clock()
            path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self._path(job_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None
