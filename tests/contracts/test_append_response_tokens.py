from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from daytona_gym.adapters.slime.sample import apply_trajectory
from daytona_gym.runtime.rollout import DaytonaTrajectory
from daytona_gym.runtime.types import OrdinalTokenizer, TrajectoryEvent
from tests.helpers import FakeSlimeSample, final_turn


class AppendAwareSample(FakeSlimeSample):
    """Mimics slime.utils.types.Sample.append_response_tokens for CPU tests."""

    def append_response_tokens(
        self,
        args=None,
        *,
        tokens=None,
        log_probs=None,
        trainable: bool = True,
        meta_info=None,
        text: str | None = None,
        update_terminal_info: bool = True,
    ) -> None:
        del args, meta_info, update_terminal_info
        tokens = list(tokens or [])
        if text is not None:
            self.response += text
        if not self.tokens:
            self.tokens = OrdinalTokenizer().encode(str(self.prompt))
        previous = self.response_length
        self.tokens += tokens
        self.response_length += len(tokens)
        if self.loss_mask is None:
            self.loss_mask = [1] * previous
        self.loss_mask += [1 if trainable else 0] * len(tokens)
        if log_probs is None and not trainable:
            log_probs = [0.0] * len(tokens)
        if log_probs is not None:
            if self.rollout_log_probs is None:
                self.rollout_log_probs = [0.0] * previous
            self.rollout_log_probs += list(log_probs)


def test_apply_trajectory_uses_append_response_tokens() -> None:
    text = final_turn("ok")
    now = datetime.now(timezone.utc)
    traj = DaytonaTrajectory(
        run_id="run_1",
        rollout_id="rollout_1",
        prompt="prompt",
        events=[
            TrajectoryEvent(
                type="generation",
                text=text,
                started_at=now,
                finished_at=now,
                token_ids=[9, 8, 7],
                log_probs=[-0.1, -0.2, -0.3],
            )
        ],
        final_response="ok",
        reward=None,
        status="completed",
        started_at=now,
        finished_at=now,
    )
    sample = AppendAwareSample(prompt="prompt", index=1)
    apply_trajectory(sample, traj, OrdinalTokenizer(), args=SimpleNamespace())
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, -0.3]
    assert sample.tokens[: len("prompt")] == OrdinalTokenizer().encode("prompt")
