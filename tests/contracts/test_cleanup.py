from __future__ import annotations

import asyncio

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_parse_user_code_error_nudges_instead_of_failing_job() -> None:
    """Hard-prompt models often emit non-object JSON; that must not kill Ray."""
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            '["not", "an", "object"]',
            tool_turn("run_tests", {"command": "pytest"}),
            final_turn("fixed"),
        ]
    )
    sample = FakeSlimeSample(prompt="p", index=42)
    args = make_args(runtime=runtime, generator=generator)

    result = await generate(args, sample, {})

    assert result.metadata["daytona"]["status"] == "completed"
    assert runtime.leaked_sandbox_ids == ()


async def test_max_tools_per_turn_caps_spam() -> None:
    from daytona_gym.telemetry.store import InMemoryTelemetryStore

    store = InMemoryTelemetryStore()
    runtime = FakeEnvironmentRuntime()
    spam = " ".join(
        '{"type":"run_tests","arguments":{"command":"python test_broken.py"}}'
        for _ in range(20)
    )
    generator = ScriptedGenerator([spam, final_turn("fixed")])
    sample = FakeSlimeSample(prompt="p", index=43)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_store=store,
        daytona_max_tools_per_turn=3,
        daytona_require_passing_tests_for_final=False,
    )

    result = await generate(args, sample, {})

    assert result.metadata["daytona"]["status"] == "completed"
    tool_spans = [span for span in store.spans if span.name.startswith("tool.")]
    assert len(tool_spans) == 3
    assert runtime.leaked_sandbox_ids == ()


def test_context_overflow_heuristic() -> None:
    from daytona_gym.runtime.rollout import _looks_like_context_overflow

    assert _looks_like_context_overflow(
        "Requested token count exceeds the model's maximum context length of 32768 tokens"
    )
    assert not _looks_like_context_overflow("sglang router request failed: connection reset")


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

    with pytest.raises(DaytonaError) as caught:
        await generate(args, sample, {})

    assert caught.value.code == ErrorCode.ROLLOUT_TIMEOUT
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

    with pytest.raises(DaytonaError) as caught:
        await generate(args, sample, {})

    assert caught.value.code == ErrorCode.TOOL_TIMEOUT
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

    with pytest.raises(DaytonaError) as caught:
        await generate(args, sample, {})

    assert caught.value.code == ErrorCode.SANDBOX_PROVISION_FAILED
    assert runtime.created_ids == ()
    assert runtime.closed_ids == ()
    assert runtime.leaked_sandbox_ids == ()
