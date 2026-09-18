from __future__ import annotations

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.traces import RecordingTracer
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_spans_close_on_success() -> None:
    runtime = FakeEnvironmentRuntime()
    tracer = RecordingTracer()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo hi"}),
            final_turn("hi"),
        ]
    )
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(runtime=runtime, generator=generator, tracer=tracer)

    await generate(args, sample, {})

    assert tracer.open_spans == ()
    names = [record.name for record in tracer.records]
    assert names[0] == "rollout"
    assert "sandbox.provision" in names
    assert "inference.generate" in names
    assert "tool.run_command" in names
    assert "sandbox.finalize" in names
    rollout = tracer.records[0]
    assert rollout.attributes["run_id"] == "run_test"
    assert rollout.attributes["rollout_id"] == "rollout_1"
    assert all(record.finished_at is not None for record in tracer.records)


async def test_spans_close_on_provision_failure() -> None:
    runtime = FakeEnvironmentRuntime(fail_create=True)
    tracer = RecordingTracer()
    generator = ScriptedGenerator([final_turn("unused")])
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(runtime=runtime, generator=generator, tracer=tracer)

    with pytest.raises(DaytonaError) as caught:
        await generate(args, sample, {})

    assert caught.value.code == ErrorCode.SANDBOX_PROVISION_FAILED
    assert tracer.open_spans == ()
    assert any(record.name == "rollout" for record in tracer.records)
    assert all(record.finished_at is not None for record in tracer.records)
