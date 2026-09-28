"""Job queue + claim API for the outbound worker agent."""

from __future__ import annotations

import json
from pathlib import Path

from daytona_gym.ingest.jobs import JobQueue
from daytona_gym.ingest.server import serve


def test_job_queue_claim_and_complete(tmp_path: Path) -> None:
    q = JobQueue(tmp_path)
    job = q.enqueue({"payload": {"recipe": {"batch_size": 1}}})
    claimed = q.claim(worker_id="w1")
    assert claimed is not None
    assert claimed["id"] == job["id"]
    assert claimed["status"] == "running"
    assert q.claim(worker_id="w2") is None
    done = q.update(job["id"], status="completed", result={"returncode": 0})
    assert done is not None
    assert done["status"] == "completed"


def test_ingest_job_http_roundtrip(tmp_path: Path) -> None:
    import threading
    import urllib.request

    server = serve(tmp_path, host="127.0.0.1", port=0, token="tok-test-12345678")
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{port}"

    def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer tok-test-12345678",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode() or "{}")

    import urllib.error

    code, job = call("POST", "/v1/jobs", {"payload": {"hello": 1}})
    assert code == 201
    assert job["status"] == "queued"
    code, empty = call("POST", "/v1/workers/claim", {"worker_id": "agent-1"})
    assert code == 200
    assert empty["id"] == job["id"]
    code, none = call("POST", "/v1/workers/claim", {"worker_id": "agent-2"})
    assert code == 200 and none.get("job") is None
    server.shutdown()
