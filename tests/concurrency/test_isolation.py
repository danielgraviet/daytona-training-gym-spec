from __future__ import annotations

import asyncio

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_concurrent_rollouts_do_not_share_environment_state() -> None:
    runtime = FakeEnvironmentRuntime()

    async def run_one(index: int, content: str) -> FakeSlimeSample:
        generator = ScriptedGenerator(
            [
                tool_turn("write_file", {"path": "note.txt", "content": content}),
                tool_turn("read_file", {"path": "note.txt"}),
                final_turn(content),
            ]
        )
        sample = FakeSlimeSample(prompt=f"task-{index}", index=index, rollout_id=index)
        args = make_args(runtime=runtime, generator=generator)
        return await generate(args, sample, {})

    results = await asyncio.gather(
        run_one(1, "alpha"),
        run_one(2, "beta"),
        run_one(3, "gamma"),
    )

    texts = [sample.response for sample in results]
    assert "alpha" in texts[0] and "beta" not in texts[0] and "gamma" not in texts[0]
    assert "beta" in texts[1] and "alpha" not in texts[1]
    assert "gamma" in texts[2] and "alpha" not in texts[2]
    assert all(sample.status is FakeSlimeSample.Status.COMPLETED for sample in results)
    assert runtime.leaked_sandbox_ids == ()
    assert len(runtime.created_ids) == 3
    assert set(runtime.closed_ids) == set(runtime.created_ids)


async def test_slow_rollout_does_not_block_fast_rollout() -> None:
    runtime = FakeEnvironmentRuntime()
    order: list[str] = []

    async def slow() -> None:
        slow_runtime = FakeEnvironmentRuntime(tool_delay_seconds=0.15)
        generator = ScriptedGenerator(
            [
                tool_turn("run_command", {"command": "echo slow"}),
                final_turn("slow-done"),
            ]
        )
        sample = FakeSlimeSample(prompt="slow", index=10)
        args = make_args(runtime=slow_runtime, generator=generator)
        result = await generate(args, sample, {})
        order.append("slow")
        assert result.status is FakeSlimeSample.Status.COMPLETED
        assert slow_runtime.leaked_sandbox_ids == ()

    async def fast() -> None:
        generator = ScriptedGenerator([final_turn("fast-done")])
        sample = FakeSlimeSample(prompt="fast", index=11)
        args = make_args(runtime=runtime, generator=generator)
        result = await generate(args, sample, {})
        order.append("fast")
        assert result.status is FakeSlimeSample.Status.COMPLETED

    await asyncio.gather(slow(), fast())
    assert order == ["fast", "slow"]
