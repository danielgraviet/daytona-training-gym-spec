from __future__ import annotations

from datetime import datetime, timezone

from daytona_gym.adapters.slime.reward import reward as slime_reward
from daytona_gym.adapters.slime.sample import apply_trajectory, infer_reward_from_trajectory
from daytona_gym.runtime.generation import GenerationResult, ScriptedGenerator
from daytona_gym.runtime.rollout import DaytonaTrajectory
from daytona_gym.runtime.types import OrdinalTokenizer, TrajectoryEvent
from tests.helpers import FakeSlimeSample, final_turn, tool_turn


def _event(
    type_: str,
    text: str,
    *,
    tool_name: str | None = None,
    ok: bool | None = None,
    exit_code: int | None = None,
    token_ids: list[int] | None = None,
    log_probs: list[float] | None = None,
) -> TrajectoryEvent:
    now = datetime.now(timezone.utc)
    return TrajectoryEvent(
        type=type_,
        text=text,
        started_at=now,
        finished_at=now,
        tool_name=tool_name,
        ok=ok,
        exit_code=exit_code,
        token_ids=token_ids,
        log_probs=log_probs,
    )


def test_apply_trajectory_prefers_engine_token_ids_and_logprobs() -> None:
    gen_text = final_turn("ok")
    token_ids = [10, 20, 30, 40]
    log_probs = [-0.1, -0.2, -0.3, -0.4]
    traj = DaytonaTrajectory(
        run_id="run_1",
        rollout_id="rollout_1",
        prompt="prompt",
        events=[
            _event(
                "generation",
                gen_text,
                token_ids=token_ids,
                log_probs=log_probs,
            )
        ],
        final_response="ok",
        reward=None,
        status="completed",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    sample = FakeSlimeSample(prompt="prompt", index=1)
    apply_trajectory(sample, traj, OrdinalTokenizer())

    assert sample.tokens == OrdinalTokenizer().encode("prompt") + token_ids
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 1, 1, 1]
    assert sample.rollout_log_probs == log_probs


def test_apply_trajectory_masks_tool_observations() -> None:
    gen = tool_turn("run_tests", {"command": "pytest"})
    obs = "\n<tool_result>...</tool_result>\n"
    traj = DaytonaTrajectory(
        run_id="run_1",
        rollout_id="rollout_1",
        prompt="p",
        events=[
            _event("generation", gen, token_ids=[1, 2], log_probs=[-0.1, -0.2]),
            _event("tool", obs, tool_name="run_tests", ok=True, exit_code=0),
            _event(
                "generation",
                final_turn("done"),
                token_ids=[3],
                log_probs=[-0.3],
            ),
        ],
        final_response="done",
        reward=None,
        status="completed",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    sample = FakeSlimeSample(prompt="p")
    apply_trajectory(sample, traj, OrdinalTokenizer())
    assert sample.loss_mask is not None
    assert sample.loss_mask.count(1) == 3
    assert sample.loss_mask.count(0) == len(OrdinalTokenizer().encode(obs))
    assert sample.reward == 1.0


def test_infer_reward_from_failed_tests() -> None:
    traj = DaytonaTrajectory(
        run_id="run_1",
        rollout_id="rollout_1",
        prompt="p",
        events=[
            _event("tool", "fail", tool_name="run_tests", ok=False, exit_code=1),
        ],
        final_response=None,
        reward=None,
        status="completed",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    assert infer_reward_from_trajectory(traj) == 0.0


async def test_slime_reward_reads_daytona_metadata() -> None:
    sample = FakeSlimeSample(prompt="p", metadata={"daytona": {"reward": 0.75}})
    assert await slime_reward(object(), sample) == 0.75


async def test_scripted_generator_result_with_logprobs_roundtrips() -> None:
    from daytona_gym.adapters.slime import generate
    from daytona_gym.runtime.fake import FakeEnvironmentRuntime
    from tests.helpers import make_args

    text = final_turn("hi")
    generator = ScriptedGenerator(
        [
            GenerationResult(
                text=text,
                token_ids=[ord(c) for c in text],
                log_probs=[-0.01] * len(text),
            )
        ]
    )
    sample = FakeSlimeSample(prompt="p", index=9)
    args = make_args(runtime=FakeEnvironmentRuntime(), generator=generator)
    await generate(args, sample, {})
    assert sample.rollout_log_probs is not None
    assert len(sample.rollout_log_probs) == sample.response_length
