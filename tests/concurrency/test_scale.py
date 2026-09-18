from __future__ import annotations

import asyncio

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.runtime.limits import LimitingEnvironmentRuntime
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_120_concurrent_rollouts_all_complete_and_clean_up() -> None:
    runtime = FakeEnvironmentRuntime()

    async def run_one(index: int) -> FakeSlimeSample:
        generator = ScriptedGenerator(
            [
                tool_turn("write_file", {"path": "note.txt", "content": str(index)}),
                tool_turn("read_file", {"path": "note.txt"}),
                final_turn(str(index)),
            ]
        )
        sample = FakeSlimeSample(prompt=f"task-{index}", index=index, rollout_id=index)
        args = make_args(runtime=runtime, generator=generator)
        return await generate(args, sample, {})

    results = await asyncio.gather(*[run_one(i) for i in range(120)])
    assert all(sample.status is FakeSlimeSample.Status.COMPLETED for sample in results)
    assert all(sample.response_length > 0 for sample in results)
    for index, sample in enumerate(results):
        assert str(index) in sample.response
    assert runtime.leaked_sandbox_ids == ()
    assert len(runtime.created_ids) == 120
    assert set(runtime.closed_ids) == set(runtime.created_ids)


async def test_one_failed_rollout_does_not_block_the_rest() -> None:
    runtime = FakeEnvironmentRuntime()

    async def run_one(index: int, *, fail: bool) -> FakeSlimeSample | DaytonaError:
        generator = ScriptedGenerator(
            [final_turn("done")] if not fail else [final_turn("unused")]
        )
        sample = FakeSlimeSample(prompt=f"task-{index}", index=index)
        overrides = {}
        if fail:
            overrides["daytona_env_metadata"] = {"fake.fail_create": "true"}
        args = make_args(runtime=runtime, generator=generator, **overrides)
        try:
            return await generate(args, sample, {})
        except DaytonaError as exc:
            return exc

    results = await asyncio.gather(
        *[run_one(i, fail=(i == 7)) for i in range(20)]
    )
    failed = results[7]
    assert isinstance(failed, DaytonaError)
    assert failed.code == ErrorCode.SANDBOX_PROVISION_FAILED
    succeeded = [sample for i, sample in enumerate(results) if i != 7]
    assert all(
        isinstance(sample, FakeSlimeSample)
        and sample.status is FakeSlimeSample.Status.COMPLETED
        for sample in succeeded
    )
    assert runtime.leaked_sandbox_ids == ()


async def test_limited_concurrent_generate_does_not_leak() -> None:
    inner = FakeEnvironmentRuntime(create_delay_seconds=0.01)
    runtime = LimitingEnvironmentRuntime(inner, max_in_flight=10)

    async def run_one(index: int) -> FakeSlimeSample:
        generator = ScriptedGenerator([final_turn(f"done-{index}")])
        sample = FakeSlimeSample(prompt=f"task-{index}", index=index)
        args = make_args(runtime=runtime, generator=generator)
        return await generate(args, sample, {})

    results = await asyncio.gather(*[run_one(i) for i in range(40)])
    assert all(sample.status is FakeSlimeSample.Status.COMPLETED for sample in results)
    assert runtime.peak_in_flight <= 10
    assert runtime.in_flight == 0
    assert inner.leaked_sandbox_ids == ()
