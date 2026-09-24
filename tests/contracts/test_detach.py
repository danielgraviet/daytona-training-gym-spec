"""CPU tests for detached remote_job spawn/status protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daytona_gym.gym.remote_job import spawn_detached_run
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def test_spawn_detached_run_reads_status(tmp_path: Path, monkeypatch) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    payload = {"detach_ready_timeout_seconds": 5}

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    written: dict = {}

    def fake_popen(cmd, stdout=None, stderr=None, env=None, start_new_session=False):  # noqa: ANN001
        status = Path(env["DAYTONA_GYM_STATUS_FILE"])
        status.write_text(
            json.dumps(
                {
                    "run_id": "run_x",
                    "telemetry_path": "/tmp/runs/run_x.jsonl",
                    "dashboard_url": "https://example.trycloudflare.com/run/run_x",
                    "pid": 4242,
                }
            ),
            encoding="utf-8",
        )
        written["status"] = status
        if stdout is not None:
            stdout.close()
        return FakeProc()

    monkeypatch.setattr("daytona_gym.gym.remote_job.subprocess.Popen", fake_popen)
    run = spawn_detached_run(job, payload)
    assert run.detached is True
    assert run.returncode is None
    assert run.training_run_id == "run_x"
    assert run.dashboard_url.endswith("/run/run_x")


def test_spawn_detached_run_surfaces_child_error(tmp_path: Path, monkeypatch) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")

    class FakeProc:
        pid = 1

        def poll(self):
            return None

    def fake_popen(cmd, stdout=None, stderr=None, env=None, start_new_session=False):  # noqa: ANN001
        Path(env["DAYTONA_GYM_STATUS_FILE"]).write_text(
            json.dumps({"error": "[platform_error] boom", "run_id": "failed"}),
            encoding="utf-8",
        )
        if stdout is not None:
            stdout.close()
        return FakeProc()

    monkeypatch.setattr("daytona_gym.gym.remote_job.subprocess.Popen", fake_popen)
    with pytest.raises(DaytonaError) as caught:
        spawn_detached_run(job, {"detach_ready_timeout_seconds": 2})
    assert caught.value.code == ErrorCode.PLATFORM_ERROR
    assert "boom" in caught.value.message
