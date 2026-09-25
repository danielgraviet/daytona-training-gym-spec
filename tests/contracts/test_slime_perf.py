"""1.2: Slime's own per-step perf numbers, parsed from Ray job output."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from daytona_gym.gym.launch import _run_slime_job
from daytona_gym.telemetry.analysis import analyze_run
from daytona_gym.telemetry.slime_perf import SlimePerfRecorder, parse_perf_line
from daytona_gym.telemetry.store import InMemoryTelemetryStore, StoredSpan

# Shape observed live on A100 (2026-09-25), prefix as Ray prints it.
LIVE = (
    "(MegatronTrainRayActor pid=4342) [2026-09-25 16:15:29] train.py:88 - "
    "{'perf/log_probs_tflops': 100.12230356866365, 'perf/actor_train_tflops': 91.20871086053022, "
    "'perf/actor_train_tok_per_s': 4851.692534923371, 'perf/step_time': 43.87778568267822, "
    "'perf/wait_time_ratio': 0.9625731916228133}"
)


def test_parses_live_line() -> None:
    perf, step = parse_perf_line(LIVE)
    assert step is None  # pid=4342 must not be mistaken for a step
    assert perf["perf/step_time"] == pytest.approx(43.8778, rel=1e-4)
    assert perf["perf/wait_time_ratio"] == pytest.approx(0.9626, rel=1e-3)


@pytest.mark.parametrize(
    ("line", "step"),
    [
        ("perf 3: {'perf/step_time': 1.0}", 3),
        ("rollout_id=7 {'perf/step_time': 1.0}", 7),
        ("[step 2] {'perf/step_time': 1.0}", 2),
    ],
)
def test_parses_explicit_step(line: str, step: int) -> None:
    assert parse_perf_line(line)[1] == step


@pytest.mark.parametrize(
    "line",
    [
        "Initializing Megatron…",
        "{'loss': 0.1}",  # no perf keys
        "{'perf/step_time': __import__('os').system('x')}",  # never eval
        "perf/step_time went up {broken",
    ],
)
def test_ignores_non_perf_and_garbage(line: str) -> None:
    assert parse_perf_line(line) is None


def test_recorder_writes_one_step_per_line_and_dedupes(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    rec = SlimePerfRecorder(path, run_id="r")
    for line in ["boot", LIVE, LIVE, LIVE.replace("43.87", "40.00"), "noise {"]:
        rec.feed(line)
    assert rec.steps_recorded == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    steps = {(r["labels"]["step"], r["labels"]["step_source"]) for r in rows}
    assert steps == {("0", "order"), ("1", "order")}
    assert {r["name"] for r in rows} >= {"slime.perf.step_time", "slime.perf.wait_time_ratio"}


def test_launcher_hands_every_output_line_to_on_line(tmp_path) -> None:
    seen: list[str] = []
    script = f"print('hello'); print({LIVE!r})"
    rc = _run_slime_job(
        [sys.executable, "-c", script], cwd=tmp_path, env=dict(os.environ), on_line=seen.append
    )
    assert rc == 0
    assert "hello" in seen and LIVE in seen


def test_analysis_splits_trainer_gap_into_train_and_overhead(tmp_path) -> None:
    """Shape of the live A100 run: ~5s rollouts inside a ~44s Slime step."""
    t0 = datetime(2026, 9, 25, tzinfo=timezone.utc)
    iso = lambda s: (t0 + timedelta(seconds=s)).isoformat()  # noqa: E731
    store = InMemoryTelemetryStore()
    for step in range(2):
        base = step * 44.0
        store.record_span(StoredSpan(name="inference.generate", started_at=iso(base), duration_seconds=2.0,
                                     attributes={"rollout_id": f"r{step}", "training_step": step}))
        store.record_span(StoredSpan(name="tool.run_tests", started_at=iso(base + 2), duration_seconds=3.0,
                                     attributes={"rollout_id": f"r{step}", "training_step": step}))
        store.record_span(StoredSpan(name="rollout", started_at=iso(base), duration_seconds=5.0,
                                     attributes={"rollout_id": f"r{step}", "training_step": step, "status": "completed"}))
    path = tmp_path / "run.jsonl"
    store.dump_jsonl(path)
    rec = SlimePerfRecorder(path, run_id="r")
    rec.feed("{'perf/step_time': 44.0, 'perf/wait_time_ratio': 0.96}")
    rec.feed("{'perf/step_time': 44.0, 'perf/wait_time_ratio': 0.95}")

    out = analyze_run(InMemoryTelemetryStore.load_jsonl(path))
    s0 = out["steps"][0]["slime"]
    assert s0["train_seconds"] == pytest.approx(44 * 0.04)
    assert s0["overhead_seconds"] == pytest.approx(44 * 0.96 - 5.0)
    totals = out["totals"]
    assert totals["trainer_source"] == "slime"
    assert totals["bottleneck"] == "overhead"
    assert set(totals["time_breakdown"]) == {"environment", "inference", "trainer", "overhead"}
    shares = sum(v["share"] for v in totals["time_breakdown"].values())
    assert shares == pytest.approx(1.0)


def test_without_slime_numbers_falls_back_to_derived_trainer() -> None:
    store = InMemoryTelemetryStore()
    store.record_span(StoredSpan(name="rollout", started_at="2026-09-25T00:00:00+00:00", duration_seconds=1.0,
                                 attributes={"rollout_id": "a", "training_step": 0, "status": "completed"}))
    totals = analyze_run(store)["totals"]
    assert totals["trainer_source"] == "derived"
    assert "overhead" not in totals["time_breakdown"]
