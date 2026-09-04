from __future__ import annotations

import json
from pathlib import Path

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.store import (
    InMemoryTelemetryStore,
    counter_by_label,
    reconstruct_rollout,
    wall_time_decomposition,
)
from daytona_gym.telemetry.traces import StoringTracer
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn

_REQUIRED_IDS = {
    "project_id",
    "run_id",
    "rollout_id",
    "sample_id",
    "sandbox_id",
    "worker_id",
    "training_step",
    "rollout_batch_id",
}


async def test_rollout_can_be_reconstructed_chronologically() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo hi"}),
            final_turn("hi"),
        ]
    )
    store = InMemoryTelemetryStore()
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_store=store,
        daytona_worker_id="gpu-1",
        daytona_training_step=12,
        daytona_rollout_batch_id="batch-a",
    )

    await generate(args, sample, {})

    timeline = reconstruct_rollout(store, "rollout_1")
    names = [step.name for step in timeline]
    assert names == [
        "rollout",
        "sandbox.provision",
        "inference.generate",
        "tool.run_command",
        "inference.generate",
        "sandbox.finalize",
    ]
    assert all(step.offset_seconds >= 0 for step in timeline)
    assert timeline[0].offset_seconds == 0.0
    assert all(left.offset_seconds <= right.offset_seconds for left, right in zip(timeline, timeline[1:]))
    assert all(_REQUIRED_IDS <= set(step.attributes) for step in timeline)
    assert timeline[0].attributes["worker_id"] == "gpu-1"
    assert timeline[0].attributes["training_step"] == 12
    assert timeline[0].attributes["status"] == "completed"
    assert timeline[0].error is None
    assert args.daytona_tracer.open_spans == ()


async def test_stored_telemetry_roundtrips_json_and_jsonl(tmp_path: Path) -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("done")])
    store = InMemoryTelemetryStore()
    sample = FakeSlimeSample(prompt="p", index=4)
    args = make_args(runtime=runtime, generator=generator, daytona_telemetry_store=store)

    await generate(args, sample, {})

    payload = json.dumps(store.to_dict())
    restored = InMemoryTelemetryStore.from_dict(json.loads(payload))
    original = [step.name for step in reconstruct_rollout(store, "rollout_4")]
    assert [step.name for step in reconstruct_rollout(restored, "rollout_4")] == original

    path = tmp_path / "run.jsonl"
    store.dump_jsonl(path)
    from_file = InMemoryTelemetryStore.load_jsonl(path)
    assert [step.name for step in reconstruct_rollout(from_file, "rollout_4")] == original
    assert from_file.metrics_named("rollout.duration_seconds")


async def test_metrics_capture_duration_and_status_counts() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo hi"}),
            final_turn("hi"),
        ]
    )
    store = InMemoryTelemetryStore()
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(runtime=runtime, generator=generator, daytona_telemetry_store=store)

    await generate(args, sample, {})

    counts = counter_by_label(store.metrics_named("rollout.count"), "status")
    assert counts["completed"] == 1.0
    assert store.metrics_named("rollout.duration_seconds")
    assert store.metrics_named("sandbox.startup_seconds")
    assert store.metrics_named("tool.duration_seconds")
    tool = store.metrics_named("tool.duration_seconds")[0]
    assert tool.labels["tool"] == "run_command"
    decomposition = wall_time_decomposition(store.spans_for_rollout("rollout_1"))
    assert decomposition["sandbox"] > 0
    assert decomposition["inference"] > 0
    assert decomposition["environment"] > 0


async def test_reward_span_is_stored_when_reward_function_is_configured() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("done")])
    store = InMemoryTelemetryStore()

    async def reward(_args: object, _sample: object) -> float:
        return 0.75

    sample = FakeSlimeSample(prompt="p", index=2)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_store=store,
        daytona_reward_function=reward,
    )

    await generate(args, sample, {})

    names = [step.name for step in reconstruct_rollout(store, "rollout_2")]
    assert names[-1] == "reward.compute"
    assert sample.reward == 0.75
    assert sample.metadata["daytona"]["reward"] == 0.75
    assert store.metrics_named("reward.duration_seconds")


async def test_reward_failure_uses_typed_error() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("done")])
    sample = FakeSlimeSample(prompt="p", index=3)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_reward_function=lambda _args, _sample: (_ for _ in ()).throw(ValueError("boom")),
    )

    with pytest.raises(DaytonaError) as exc_info:
        await generate(args, sample, {})

    assert exc_info.value.code is ErrorCode.REWARD_FAILED
    assert runtime.leaked_sandbox_ids == ()


async def test_concurrent_rollouts_are_isolated_in_the_store() -> None:
    import asyncio

    runtime = FakeEnvironmentRuntime()
    store = InMemoryTelemetryStore()

    async def run_one(index: int) -> None:
        generator = ScriptedGenerator([final_turn(str(index))])
        sample = FakeSlimeSample(prompt=f"p-{index}", index=index)
        args = make_args(
            runtime=runtime,
            generator=generator,
            daytona_telemetry_store=store,
        )
        await generate(args, sample, {})

    await asyncio.gather(*[run_one(i) for i in range(8)])

    for index in range(8):
        timeline = reconstruct_rollout(store, f"rollout_{index}")
        assert [step.name for step in timeline][0] == "rollout"
        assert {step.attributes["rollout_id"] for step in timeline} == {f"rollout_{index}"}
        assert {step.attributes["sample_id"] for step in timeline} == {str(index)}


async def test_failed_provision_timeline_records_the_error() -> None:
    runtime = FakeEnvironmentRuntime(fail_create=True)
    store = InMemoryTelemetryStore()
    generator = ScriptedGenerator([final_turn("unused")])
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(runtime=runtime, generator=generator, daytona_telemetry_store=store)

    await generate(args, sample, {})

    timeline = reconstruct_rollout(store, "rollout_1")
    names = [step.name for step in timeline]
    assert names[0] == "rollout"
    assert "sandbox.provision" in names
    provision = next(step for step in timeline if step.name == "sandbox.provision")
    assert provision.error == "DaytonaError"
    assert timeline[0].attributes["status"] == "failed"
    counts = counter_by_label(store.metrics_named("rollout.count"), "status")
    assert counts["failed"] == 1.0


def test_storing_tracer_redacts_secret_attributes() -> None:
    store = InMemoryTelemetryStore()
    tracer = StoringTracer(store)
    with tracer.span("rollout", run_id="run_1", api_key="super-secret"):
        pass
    assert store.spans[0].attributes["api_key"] == "***"
    assert store.spans[0].attributes["run_id"] == "run_1"


def test_store_drops_oldest_spans_when_capped() -> None:
    store = InMemoryTelemetryStore(max_spans=3, max_metrics=10, max_rollouts=None)
    tracer = StoringTracer(store)
    for index in range(5):
        with tracer.span("rollout", rollout_id="r1", n=index):
            pass
    assert len(store.spans) == 3
    assert store.dropped_spans == 2
    assert [span.attributes["n"] for span in store.spans] == [2, 3, 4]


def test_store_drops_oldest_rollout_as_a_unit() -> None:
    store = InMemoryTelemetryStore(max_spans=100, max_metrics=100, max_rollouts=2)
    tracer = StoringTracer(store)
    for rollout_id in ("a", "b", "c"):
        with tracer.span("rollout", rollout_id=rollout_id):
            pass
        with tracer.span("tool.run_command", rollout_id=rollout_id):
            pass
    assert reconstruct_rollout(store, "a") == ()
    assert [step.name for step in reconstruct_rollout(store, "b")] == [
        "rollout",
        "tool.run_command",
    ]
    assert [step.name for step in reconstruct_rollout(store, "c")] == [
        "rollout",
        "tool.run_command",
    ]
    assert store.dropped_rollouts == 1
    assert store.dropped_spans == 2


def test_store_drops_oldest_metrics_when_capped() -> None:
    from daytona_gym.telemetry.metrics import StoringMetrics

    store = InMemoryTelemetryStore(max_spans=100, max_metrics=2, max_rollouts=None)
    metrics = StoringMetrics(store)
    for _ in range(5):
        metrics.increment("rollout.count", status="completed")
    assert len(store.metrics) == 2
    assert store.dropped_metrics == 3
