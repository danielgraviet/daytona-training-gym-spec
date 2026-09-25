"""Modal-shaped TrainingRun.wait / result completion."""

from __future__ import annotations

from pathlib import Path

from daytona_gym.gym.run import TrainingRun
from daytona_gym.telemetry.progress import write_progress


def test_result_prints_training_complete(tmp_path: Path, capsys) -> None:
    write_progress(
        tmp_path,
        "run_done",
        phase="completed",
        message="Training complete",
        status="completed",
        returncode=0,
    )
    run = TrainingRun(
        run_id="run_done",
        telemetry_path=str(tmp_path / "run_done.jsonl"),
        command=[],
        env={},
        runtime_env={},
        dry_run=False,
        detached=True,
        status="running",
        dashboard_url=None,
    )
    out = run.result()
    assert out is run
    assert run.status == "completed"
    assert run.returncode == 0
    text = capsys.readouterr().out
    assert "Training complete: run_done" in text


def test_result_prints_training_failed(tmp_path: Path, capsys) -> None:
    write_progress(
        tmp_path,
        "run_bad",
        phase="failed",
        message="[platform_error] boom",
        status="failed",
        returncode=2,
    )
    run = TrainingRun(
        run_id="run_bad",
        telemetry_path=str(tmp_path / "run_bad.jsonl"),
        command=[],
        env={},
        runtime_env={},
        dry_run=False,
        detached=True,
        status="running",
    )
    run.result()
    assert run.status == "failed"
    text = capsys.readouterr().out
    assert "Training failed: run_bad" in text


def test_wait_emits_stage_lines(capsys) -> None:
    run = TrainingRun(
        run_id="run_stage",
        telemetry_path="/tmp/missing/run_stage.jsonl",
        command=[],
        env={},
        runtime_env={},
        dry_run=False,
        detached=True,
        status="running",
    )
    snaps = [
        {
            "phase": "ray_start",
            "message": "Starting Ray head…",
            "status": "running",
            "done": False,
        },
        {
            "phase": "ray_start",
            "message": "Starting Ray head…",
            "status": "running",
            "done": False,
        },
        {
            "phase": "completed",
            "message": "Training complete",
            "status": "completed",
            "done": True,
            "returncode": 0,
        },
    ]

    def fake_poll() -> dict | None:
        return snaps.pop(0) if snaps else {
            "phase": "completed",
            "message": "Training complete",
            "status": "completed",
            "done": True,
            "returncode": 0,
        }

    run._poll_snapshot = fake_poll  # type: ignore[method-assign]
    run.wait(poll_interval=0.01)
    text = capsys.readouterr().out
    assert "[ray_start] Starting Ray head…" in text
    assert "Starting Ray head…" in text
    assert run.status == "completed"


def test_live_api_url_from_deep_link() -> None:
    run = TrainingRun(
        run_id="abc",
        telemetry_path="/tmp/runs/abc.jsonl",
        command=[],
        env={},
        runtime_env={},
        dashboard_url="https://foo.trycloudflare.com/run/abc",
    )
    assert run._live_api_url() == "https://foo.trycloudflare.com/api/runs/abc/live"
