from __future__ import annotations

import json

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.errors import ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_successful_rollout_returns_completed_sample() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "echo hello"}),
            final_turn("hello"),
        ]
    )
    sample = FakeSlimeSample(prompt="fix the parser", index=3, rollout_id=9)
    args = make_args(runtime=runtime, generator=generator)

    result = await generate(args, sample, {"temperature": 0.0})

    assert result is sample
    assert sample.status is FakeSlimeSample.Status.COMPLETED
    assert sample.rollout_id == 9
    assert sample.index == 3
    assert sample.response_length == len(sample.loss_mask or [])
    assert sample.tokens[: len("fix the parser")] == [ord(c) for c in "fix the parser"]
    assert 1 in (sample.loss_mask or [])
    assert 0 in (sample.loss_mask or [])
    assert "hello" in sample.response
    metadata = sample.metadata["daytona"]
    assert metadata["run_id"] == "run_test"
    assert metadata["rollout_id"] == "rollout_9"
    assert metadata["sandbox_id"]
    assert metadata["error_code"] is None
    assert runtime.leaked_sandbox_ids == ()
    assert runtime.closed_ids == runtime.created_ids


async def test_tool_nonzero_exit_is_observation_not_hard_failure() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_command", {"command": "false"}),
            final_turn("command failed"),
        ]
    )
    sample = FakeSlimeSample(prompt="run tests", index=1)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.COMPLETED
    assert 'ok="false"' in sample.response
    assert 'exit_code="1"' in sample.response
    assert runtime.leaked_sandbox_ids == ()


async def test_sandbox_creation_failure_maps_to_failed_sample() -> None:
    runtime = FakeEnvironmentRuntime(fail_create=True)
    generator = ScriptedGenerator([final_turn("unused")])
    sample = FakeSlimeSample(prompt="task", index=2)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.FAILED
    assert sample.metadata["daytona"]["error_code"] == ErrorCode.SANDBOX_PROVISION_FAILED
    assert runtime.created_ids == ()
    assert runtime.leaked_sandbox_ids == ()


async def test_identifiers_are_preserved_on_the_slime_sample() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("done")])
    sample = FakeSlimeSample(prompt="p", index=42, rollout_id=7)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert sample.index == 42
    assert sample.rollout_id == 7
    assert sample.metadata["daytona"]["sample_id"] == "42"
    assert sample.metadata["daytona"]["rollout_id"] == "rollout_7"


async def test_chat_prompt_is_flattened() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("ok")])
    sample = FakeSlimeSample(
        prompt=[{"role": "user", "content": "fix it"}],
        index=1,
    )
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.COMPLETED
    assert sample.metadata["daytona"]["status"] == "completed"


async def test_malformed_tool_name_is_user_code_error() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([json.dumps({"type": "tool", "name": "explode", "arguments": {}})])
    sample = FakeSlimeSample(prompt="p", index=1)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.FAILED
    assert sample.metadata["daytona"]["error_code"] == ErrorCode.USER_CODE_ERROR
    assert runtime.leaked_sandbox_ids == ()
