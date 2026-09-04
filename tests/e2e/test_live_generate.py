from __future__ import annotations

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.traces import RecordingTracer
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


@pytest.mark.e2e
async def test_live_generate_returns_completed_sample_and_cleans_up() -> None:
    runtime = DaytonaEnvironmentRuntime(ephemeral=True, create_timeout_seconds=120)
    tracer = RecordingTracer()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo live-generate"}),
            tool_turn("write_file", {"path": "answer.txt", "content": "ok\n"}),
            tool_turn("read_file", {"path": "answer.txt"}),
            final_turn("ok"),
        ]
    )
    sample = FakeSlimeSample(
        prompt="write a file, read it, then finish",
        index=1,
        rollout_id=1,
    )
    args = make_args(
        runtime=runtime,
        generator=generator,
        tracer=tracer,
        daytona_image=None,
        daytona_run_id="e2e-generate",
        daytona_sandbox_timeout_seconds=120,
        daytona_tool_timeout_seconds=30,
    )
    try:
        result = await generate(args, sample, {"temperature": 0.0})
        assert result is sample
        assert sample.status is FakeSlimeSample.Status.COMPLETED
        assert sample.rollout_id == 1
        assert sample.index == 1
        assert sample.response_length == len(sample.loss_mask or [])
        assert 1 in (sample.loss_mask or [])
        assert 0 in (sample.loss_mask or [])
        assert "live-generate" in sample.response
        assert "ok" in sample.response
        metadata = sample.metadata["daytona"]
        assert metadata["run_id"] == "e2e-generate"
        assert metadata["sandbox_id"]
        assert metadata["error_code"] is None
        assert metadata["status"] == "completed"
        assert tracer.open_spans == ()
        names = [record.name for record in tracer.records]
        assert "rollout" in names
        assert "sandbox.provision" in names
        assert "inference.generate" in names
        assert "tool.run_command" in names
        assert "tool.write_file" in names
        assert "tool.read_file" in names
        assert "sandbox.finalize" in names
    finally:
        await runtime.aclose()
        assert runtime.leaked_sandbox_ids == ()
