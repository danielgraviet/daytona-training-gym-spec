from __future__ import annotations

import asyncio

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.errors import ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_rollout_timeout_aborts_and_cleans_up() -> None:
    runtime = FakeEnvironmentRuntime(tool_delay_seconds=0.2)
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo slow"}),
            final_turn("done"),
        ]
    )
    sample = FakeSlimeSample(prompt="slow", index=1)
    args = make_args(runtime=runtime, generator=generator, daytona_timeout_seconds=0.05)

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.ABORTED
    assert sample.metadata["daytona"]["error_code"] == ErrorCode.ROLLOUT_TIMEOUT
    assert runtime.leaked_sandbox_ids == ()
    assert runtime.closed_ids == runtime.created_ids


async def test_tool_timeout_aborts_and_cleans_up() -> None:
    runtime = FakeEnvironmentRuntime(tool_delay_seconds=0.2)
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo slow"}),
            final_turn("done"),
        ]
    )
    sample = FakeSlimeSample(prompt="slow", index=1)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_tool_timeout_seconds=0.05,
    )

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.ABORTED
    assert sample.metadata["daytona"]["error_code"] == ErrorCode.TOOL_TIMEOUT
    assert runtime.leaked_sandbox_ids == ()


async def test_cancellation_during_tool_cleans_up() -> None:
    runtime = FakeEnvironmentRuntime(hang_on_execute=True)
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo hang"}),
            final_turn("done"),
        ]
    )
    sample = FakeSlimeSample(prompt="hang", index=1)
    args = make_args(runtime=runtime, generator=generator)

    task = asyncio.create_task(generate(args, sample, {}))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sample.status is FakeSlimeSample.Status.ABORTED
    assert sample.metadata["daytona"]["cancelled"] is True
    assert runtime.leaked_sandbox_ids == ()
    assert runtime.closed_ids == runtime.created_ids


async def test_cancellation_during_create_does_not_leak() -> None:
    runtime = FakeEnvironmentRuntime(hang_on_create=True)
    generator = ScriptedGenerator([final_turn("unused")])
    sample = FakeSlimeSample(prompt="hang", index=1)
    args = make_args(runtime=runtime, generator=generator)

    task = asyncio.create_task(generate(args, sample, {}))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime.leaked_sandbox_ids == ()
    assert runtime.created_ids == ()


async def test_provision_failure_cleanup() -> None:
    runtime = FakeEnvironmentRuntime(fail_create=True)
    generator = ScriptedGenerator([final_turn("unused")])
    sample = FakeSlimeSample(prompt="x", index=1)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert runtime.created_ids == ()
    assert runtime.closed_ids == ()
    assert runtime.leaked_sandbox_ids == ()
