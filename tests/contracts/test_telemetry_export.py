from __future__ import annotations

import queue
import threading
from pathlib import Path

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.export import AsyncJsonlExporter
from daytona_gym.telemetry.metrics import StoringMetrics
from daytona_gym.telemetry.store import InMemoryTelemetryStore, reconstruct_rollout
from daytona_gym.telemetry.traces import StoringTracer
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


def test_exporter_drops_when_queue_is_full() -> None:
    exporter = AsyncJsonlExporter.__new__(AsyncJsonlExporter)
    exporter._queue = queue.Queue(maxsize=1)
    exporter._dropped = 0
    exporter._drop_lock = threading.Lock()
    exporter._queue.put_nowait({"type": "span"})
    exporter.emit({"type": "span"})
    assert exporter.dropped == 1


def test_trace_sample_rate_zero_keeps_failures_and_metrics() -> None:
    store = InMemoryTelemetryStore(
        max_spans=100,
        max_metrics=100,
        max_rollouts=None,
        trace_sample_rate=0.0,
    )
    tracer = StoringTracer(store)
    metrics = StoringMetrics(store)
    with tracer.span("rollout", rollout_id="ok", status="completed"):
        pass
    with tracer.span("rollout", rollout_id="bad") as span:
        span.set_attribute("status", "failed")
    metrics.increment("rollout.count", status="failed")
    assert reconstruct_rollout(store, "ok") == ()
    assert [step.attributes["rollout_id"] for step in reconstruct_rollout(store, "bad")] == ["bad"]
    assert store.skipped_spans >= 1
    assert store.metrics_named("rollout.count")


async def test_jsonl_export_preserves_rollout_after_memory_eviction(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo hi"}),
            final_turn("hi"),
        ]
    )
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
        daytona_telemetry_max_rollouts=1,
    )
    try:
        await generate(args, sample, {})
        second = FakeSlimeSample(prompt="p", index=2)
        args.daytona_generator = ScriptedGenerator([final_turn("done")])
        await generate(args, second, {})
        args.daytona_telemetry_store.flush()

        assert reconstruct_rollout(args.daytona_telemetry_store, "rollout_1") == ()
        from_disk = InMemoryTelemetryStore.load_jsonl(path)
        names = [step.name for step in reconstruct_rollout(from_disk, "rollout_1")]
        assert names[0] == "rollout"
        assert "sandbox.provision" in names
        assert "tool.run_command" in names
        assert "sandbox.finalize" in names
    finally:
        args.daytona_telemetry_store.close()


def test_jsonl_rotates_when_file_exceeds_max_bytes(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    exporter = AsyncJsonlExporter(path, max_bytes=120, max_files=2, batch_size=1)
    try:
        for index in range(30):
            exporter.emit(
                {
                    "type": "metric",
                    "name": "rollout.count",
                    "kind": "counter",
                    "value": 1.0,
                    "labels": {"status": "completed"},
                    "recorded_at": "",
                    "seq": index,
                }
            )
        exporter.flush()
        assert path.exists()
        assert Path(f"{path}.1").exists()
        assert not Path(f"{path}.2").exists()
    finally:
        exporter.close()
