"""Attached on-box launch must open the dash before training and always record
a terminal status (regression: dash said "running" after the job succeeded)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from daytona_gym.gym import config as config_mod
from daytona_gym.gym.run import TrainingRun
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.telemetry import dashboard_data
from daytona_gym.telemetry.progress import read_progress
from tests.contracts.test_worker import _config


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch):
    c = _config(tmp_path)
    c.compute = c.resolved_compute()
    c.model = None  # skip HF download/convert
    c.telemetry_path = tmp_path / "runs" / "run_local.jsonl"
    monkeypatch.setattr(config_mod.TrainConfig, "validate", lambda self, **_: None)
    events: list[str] = []

    def fake_open(self, **_):
        events.append("dash_open")
        self.dashboard_url = f"http://dash/run/{self.run_id}"
        self._dashboard = object()
        return self.dashboard_url

    monkeypatch.setattr(TrainingRun, "open", fake_open)
    monkeypatch.setattr(TrainingRun, "result", lambda self, **_: self)
    monkeypatch.setattr(TrainingRun, "close_dashboard", lambda self: events.append("dash_close"))
    return c, events


def test_success_writes_completed_and_opens_dash_first(cfg, monkeypatch) -> None:
    c, events = cfg

    def fake_execute(plan, *, on_phase=None, **_):
        events.append("train")
        on_phase("ray_start", "Submitting Slime Ray job…")
        prog = read_progress(plan.telemetry_path.parent, plan.telemetry_path.stem)
        assert prog["status"] == "running" and prog["phase"] == "ray_start"
        return TrainingRun(
            run_id=plan.run_id,
            telemetry_path=str(plan.telemetry_path),
            command=[],
            env={},
            runtime_env={},
            returncode=0,
        )

    monkeypatch.setattr(config_mod, "execute_plan", fake_execute)
    run = c.launch(open=True)

    assert events[:2] == ["dash_open", "train"]
    assert run.dashboard_url == "http://dash/run/run_worker_test"
    prog = read_progress(Path(run.telemetry_path).parent, "run_local")
    assert prog["status"] == "completed" and prog["done"] is True
    assert isinstance(prog["elapsed_s"], float)


def test_failure_writes_failed_and_closes_dash(cfg, monkeypatch) -> None:
    c, events = cfg

    def boom(plan, **_):
        raise DaytonaError(ErrorCode.PLATFORM_ERROR, "ray exploded")

    monkeypatch.setattr(config_mod, "execute_plan", boom)
    with pytest.raises(DaytonaError):
        c.launch(open=True)
    prog = read_progress(Path(c.telemetry_path).parent, "run_local")
    assert prog["status"] == "failed"
    assert "platform_error" in prog["message"] and "ray exploded" in prog["message"]
    assert "dash_close" in events


def test_old_runs_without_progress_go_stale(tmp_path: Path) -> None:
    path = tmp_path / "run_old.jsonl"
    path.write_text("{}\n")
    assert dashboard_data.infer_status_without_progress(path, has_rollouts=True) == "running"
    old = time.time() - dashboard_data.STALE_AFTER_SECONDS - 5
    os.utime(path, (old, old))
    assert dashboard_data.infer_status_without_progress(path, has_rollouts=True) == "stale"
    assert dashboard_data.infer_status_without_progress(path, has_rollouts=False) == "pending"
