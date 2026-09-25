"""Step-level analytics over synthetic span timelines (no Slime, no Daytona)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytona_gym.telemetry.analysis import (
    analyze_run,
    gaps_in,
    percentile,
    union_seconds,
)
from daytona_gym.telemetry.run_meta import append_run_meta
from daytona_gym.telemetry.store import InMemoryTelemetryStore, MetricSample, StoredSpan

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _iso(sec: float) -> str:
    return (T0 + timedelta(seconds=sec)).isoformat()


def _span(name: str, start: float, dur: float, rid: str, **attrs) -> StoredSpan:
    return StoredSpan(
        name=name,
        started_at=_iso(start),
        finished_at=_iso(start + dur),
        duration_seconds=dur,
        attributes={"rollout_id": rid, **attrs},
    )


def _rollout(store, rid, start, parts, *, step=None, status="completed"):
    """parts: list of (span_name, duration) laid end to end from ``start``."""
    t = start
    common = {} if step is None else {"training_step": step}
    for name, dur in parts:
        store.record_span(_span(name, t, dur, rid, **common))
        t += dur
    store.record_span(_span("rollout", start, t - start, rid, status=status, **common))
    return t


def test_interval_helpers() -> None:
    assert union_seconds([(0, 2), (1, 3), (5, 6)]) == 4
    assert union_seconds([(0, 10)], clip=(2, 4)) == 2
    assert gaps_in((0, 10), [(1, 2), (1.5, 4), (8, 12)]) == [(0, 1), (4, 8)]
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) == 0.0


def test_env_bound_step_with_straggler_and_cost() -> None:
    store = InMemoryTelemetryStore()
    # Step 0: A is quick; B has a 20s test run → GPU idles waiting on the sandbox.
    _rollout(store, "A", 0, [("sandbox.provision", 2), ("inference.generate", 3), ("tool.run_tests", 1), ("inference.generate", 2)], step=0)
    _rollout(store, "B", 0, [("sandbox.provision", 2), ("inference.generate", 3), ("tool.run_tests", 20), ("inference.generate", 1)], step=0)
    # Step 1 starts after a 10s train phase (B ends at 26).
    _rollout(store, "C", 36, [("inference.generate", 4)], step=1, status="failed")
    store.record_metric(
        MetricSample(name="gpu.utilization", kind="gauge", value=5.0, labels={}, recorded_at=_iso(15))
    )

    out = analyze_run(store, gpu_cost_per_hour=3.0, num_gpus=2)
    s0, s1 = out["steps"]

    assert s0["step"] == "0" and s0["n_rollouts"] == 2
    assert s0["rollout_phase_seconds"] == pytest.approx(26)
    # Inference busy: [2,5] ∪ [6,8] ∪ [25,26] = 6s → env wait 20s.
    assert s0["inference_busy_seconds"] == pytest.approx(6)
    assert s0["env_wait_seconds"] == pytest.approx(20)
    assert s0["bound"] == "environment"
    assert s0["critical_rollout_id"] == "B"
    assert s0["straggler_tax_seconds"] == pytest.approx(26 - 17)  # median of {8, 26}
    assert s0["train_phase_seconds"] == pytest.approx(10)
    assert s0["gpu_util_mean_during_env_wait"] == pytest.approx(5.0)

    assert s1["bound"] == "inference" and s1["train_phase_seconds"] is None

    totals = out["totals"]
    # environment = step-0 idle 20s; inference = 6s + step-1 4s; trainer = 10s gap
    assert totals["time_breakdown"]["environment"]["seconds"] == pytest.approx(20)
    assert totals["time_breakdown"]["inference"]["seconds"] == pytest.approx(10)
    assert totals["time_breakdown"]["trainer"]["seconds"] == pytest.approx(10)
    assert totals["bottleneck"] == "environment"
    assert totals["bottleneck_share"] == pytest.approx(0.5)
    assert totals["n_failed"] == 1
    assert totals["failed_waste_seconds"] == pytest.approx(4)
    assert totals["sandbox_provision"]["n"] == 2
    assert totals["tools"]["run_tests"]["n"] == 2

    cost = out["cost"]
    assert cost["run_cost"] == pytest.approx(40 / 3600 * 6)
    assert cost["idle_gpu_cost"] == pytest.approx(20 / 3600 * 6)
    assert "env_wait_seconds" in out["definitions"]


def test_missing_step_ids_are_derived_offline() -> None:
    store = InMemoryTelemetryStore()
    _rollout(store, "a", 0, [("inference.generate", 2)])
    _rollout(store, "b", 0.5, [("inference.generate", 2)])
    _rollout(store, "c", 10, [("inference.generate", 1)])
    out = analyze_run(store)
    assert [s["n_rollouts"] for s in out["steps"]] == [2, 1]
    assert {s["training_step_source"] for s in out["steps"]} == {"derived"}
    assert out["cost"] is None


def test_cost_comes_from_run_meta(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    store = InMemoryTelemetryStore()
    _rollout(store, "a", 0, [("tool.run_tests", 3600)], step=0)
    store.dump_jsonl(path)
    append_run_meta(path, run_id="r", gpu_cost_per_hour=2.5, num_gpus=4)

    out = analyze_run(InMemoryTelemetryStore.load_jsonl(path))
    assert out["cost"]["num_gpus"] == 4
    assert out["cost"]["idle_gpu_cost"] == pytest.approx(10.0)
    assert out["cost"]["idle_fraction"] == pytest.approx(1.0)


def test_reused_rollout_ids_across_steps_stay_separate() -> None:
    store = InMemoryTelemetryStore()
    _rollout(store, "rollout_0", 0, [("inference.generate", 2)], step=0)
    _rollout(store, "rollout_0", 10, [("inference.generate", 3)], step=1)
    out = analyze_run(store)
    assert [s["n_rollouts"] for s in out["steps"]] == [1, 1]
    assert [round(r["wall"]) for r in out["rollouts"]] == [2, 3]


def test_trainer_bound_run_like_live_a100() -> None:
    """Shape of run_b17 on A100: short rollouts, long derived train gaps."""
    store = InMemoryTelemetryStore()
    t = 0.0
    for step in range(4):
        _rollout(store, f"r{step}", t, [("sandbox.provision", 1), ("inference.generate", 2), ("tool.run_tests", 1)], step=step)
        t += 4 + 30  # 30s train phase between steps
    totals = analyze_run(store)["totals"]
    assert totals["bottleneck"] == "trainer"
    assert totals["rollout_bound"] == "environment"  # 2 of 4s idle → >= half
    assert totals["bottleneck_share"] == pytest.approx(90 / (8 + 8 + 90))
