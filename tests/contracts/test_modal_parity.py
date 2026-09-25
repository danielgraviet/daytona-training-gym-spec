"""Contract tests for Modal parity surface (dash charts, CLI, harbor, models)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daytona_gym.gym.dataset import HuggingFaceDataset, PromptJsonlDataset
from daytona_gym.gym.harbor import HarborBackend, HarborRecipe
from daytona_gym.gym.models import Qwen25_7B, Qwen25_14B
from daytona_gym.gym.recipe import CodingRecipe, Qwen25_7B_Recipe
from daytona_gym.gym.run_cli import list_runs, run_status, stop_run
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.telemetry import dashboard_data as data
from daytona_gym.telemetry.dashboard import spa_available
from daytona_gym.telemetry.gpu_metrics import sample_nvidia_smi
from daytona_gym.telemetry.progress import write_progress
from daytona_gym.telemetry.store import InMemoryTelemetryStore, MetricSample, StoredSpan


def _write_mini_run(path: Path) -> None:
    store = InMemoryTelemetryStore()
    store.record_span(
        StoredSpan(
            name="rollout",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:02+00:00",
            duration_seconds=2.0,
            attributes={"rollout_id": "rollout_0", "status": "completed", "reward": 1.0},
        )
    )
    store.record_span(
        StoredSpan(
            name="tool.run_tests",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            attributes={"rollout_id": "rollout_0", "tool": "run_tests"},
        )
    )
    store.record_metric(
        MetricSample(
            name="gpu.utilization",
            kind="gauge",
            value=55.0,
            labels={"gpu": "0"},
            recorded_at="2026-01-01T00:00:01+00:00",
        )
    )
    store.dump_jsonl(path)


def test_spa_assets_built() -> None:
    assert spa_available(), "dashboard_static/index.html missing — npm run build"


def test_run_charts_and_list_status(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    path = runs / "demo.jsonl"
    _write_mini_run(path)
    write_progress(runs, "demo", phase="completed", message="done", status="completed")
    charts = data.run_charts(path)
    assert charts["n_rollouts"] == 1
    assert charts["reward"][0]["y"] == 1.0
    assert charts["status_histogram"].get("completed") == 1
    assert charts["gpu_utilization"]
    items = data.list_run_files(runs)
    assert items[0]["stem"] == "demo"
    assert items[0]["run_status"] == "completed"


def test_run_cli_status_stop(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    write_progress(runs, "r1", phase="ray_start", message="boot", status="running")
    info = run_status("r1", runs)
    assert info["run_status"] == "running"
    out = stop_run("r1", runs)
    assert out["cancelled"] is True
    assert run_status("r1", runs)["run_status"] == "failed"
    assert list_runs(runs)


def test_harbor_backend_raises() -> None:
    from daytona_gym import TrainConfig

    cfg = TrainConfig(
        dataset=PromptJsonlDataset(path="/tmp/missing.jsonl"),
        recipe=CodingRecipe(),
        backend="harbor",
        harbor=HarborRecipe(task_name="tb"),
    )
    with pytest.raises(DaytonaError) as exc:
        cfg.launch(dry_run=True)
    assert exc.value.code == ErrorCode.USER_CODE_ERROR


def test_harbor_backend_direct() -> None:
    with pytest.raises(DaytonaError):
        HarborBackend().launch()


def test_model_presets() -> None:
    assert "7B" in Qwen25_7B().name
    assert Qwen25_14B().num_gpus == 2
    assert Qwen25_7B_Recipe().max_response_len == 1024


def test_hf_dataset_materialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDS(list):
        pass

    rows = FakeDS(
        [
            {"prompt": "fix it", "label": "ok"},
            {"prompt": "again", "label": "ok"},
        ]
    )

    def fake_load_dataset(*_a, **_k):
        return rows

    monkeypatch.setitem(__import__("sys").modules, "datasets", type("m", (), {"load_dataset": staticmethod(fake_load_dataset)})())
    # Force import path used inside materialize
    import daytona_gym.gym.dataset as ds_mod

    monkeypatch.setattr(
        "builtins.__import__",
        __import__,
    )

    out = tmp_path / "out.jsonl"
    # Call materialize with patched import inside the method via injecting module
    import sys

    sys.modules["datasets"] = type(
        "mod",
        (),
        {"load_dataset": staticmethod(fake_load_dataset)},
    )()

    helper = HuggingFaceDataset(repo_id="fake/repo", out_path=out, limit=1)
    prompt_ds = helper.materialize()
    text = Path(prompt_ds.path).read_text(encoding="utf-8")
    assert "fix it" in text
    assert len(text.strip().splitlines()) == 1


def test_sample_nvidia_smi_returns_list() -> None:
    assert isinstance(sample_nvidia_smi(), list)


def test_create_pod_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from daytona_gym.gym.providers.runpod import create_pod

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(DaytonaError) as exc:
        create_pod(wait_running=False)
    assert exc.value.code == ErrorCode.USER_CODE_ERROR
