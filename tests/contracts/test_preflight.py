from __future__ import annotations

import pytest

from daytona_gym.adapters.slime.generate import _raise_if_unusable
from daytona_gym.preflight import check_daytona, main as preflight_main
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.rollout import DaytonaTrajectory
from tests.helpers import FakeSlimeSample
from datetime import datetime, timezone


def test_raise_if_unusable_on_failed_trajectory() -> None:
    sample = FakeSlimeSample(prompt="x", index=1)
    now = datetime.now(timezone.utc)
    traj = DaytonaTrajectory(
        run_id="r",
        rollout_id="rollout_1",
        prompt="x",
        events=[],
        final_response=None,
        reward=None,
        status="failed",
        started_at=now,
        finished_at=now,
        error_code=str(ErrorCode.SANDBOX_PROVISION_FAILED),
        error_message="unauthorized",
    )
    with pytest.raises(DaytonaError) as caught:
        _raise_if_unusable(sample, traj)
    assert caught.value.code == ErrorCode.SANDBOX_PROVISION_FAILED
    assert "unauthorized" in caught.value.message
    assert sample.remove_sample is True


def test_raise_if_unusable_noop_on_completed() -> None:
    sample = FakeSlimeSample(prompt="x", index=1)
    now = datetime.now(timezone.utc)
    traj = DaytonaTrajectory(
        run_id="r",
        rollout_id="rollout_1",
        prompt="x",
        events=[],
        final_response="ok",
        reward=1.0,
        status="completed",
        started_at=now,
        finished_at=now,
    )
    _raise_if_unusable(sample, traj)


@pytest.mark.asyncio
async def test_preflight_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    with pytest.raises(DaytonaError) as caught:
        await check_daytona(api_key=None)
    assert "DAYTONA_API_KEY" in caught.value.message


def test_preflight_main_missing_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    assert preflight_main([]) == 1
    err = capsys.readouterr().err
    assert "preflight FAILED" in err
