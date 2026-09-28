"""Token budget truncates rollouts before the trainer sees a mega-sample."""

from __future__ import annotations

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import GenerationResult, ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args


async def test_max_total_tokens_truncates_rollout() -> None:
    runtime = FakeEnvironmentRuntime()
    # Explicit token_ids make the budget deterministic (no char heuristic).
    fat = GenerationResult(text='{"type":"tool","name":"run_command","arguments":{"command":"x"}}', token_ids=list(range(80)))
    generator = ScriptedGenerator(
        [
            fat,
            fat,
            final_turn("never"),
        ]
    )
    sample = FakeSlimeSample(prompt="p", index=7)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_max_total_tokens=100,
        daytona_max_turns=8,
        daytona_require_passing_tests_for_final=False,
        daytona_stdout_limit=256,
    )
    result = await generate(args, sample, {})
    assert result.metadata["daytona"]["status"] == "truncated"
    assert runtime.leaked_sandbox_ids == ()
