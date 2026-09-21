"""Live Daytona: tool timeout must classify as tool_timeout, not tool_failed."""

from __future__ import annotations

import pytest

from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import EnvironmentSpec, ToolAction, ToolName


@pytest.mark.e2e
async def test_live_tool_stall_is_tool_timeout_not_tool_failed() -> None:
    """Real sandbox, real SDK: short budget on a long sleep.

    Regression for H100 edge tool_stall where Daytona killed the exec but we
    labeled it tool_failed.
    """
    runtime = DaytonaEnvironmentRuntime(ephemeral=True, create_timeout_seconds=120)
    env = None
    try:
        env = await runtime.create(
            EnvironmentSpec(
                metadata={
                    "run_id": "e2e",
                    "rollout_id": "live-tool-stall",
                    "project_id": "daytona-gym",
                }
            )
        )
        with pytest.raises(DaytonaError) as caught:
            await runtime.execute(
                env,
                ToolAction(
                    name=ToolName.RUN_TESTS,
                    arguments={
                        "command": "python -c 'import time; time.sleep(120)'",
                    },
                    timeout_seconds=5,
                ),
            )
        assert caught.value.code is ErrorCode.TOOL_TIMEOUT, (
            f"expected tool_timeout, got {caught.value.code}: {caught.value.message}"
        )
    finally:
        if env is not None:
            await runtime.close(env)
        await runtime.aclose()
        assert runtime.leaked_sandbox_ids == ()
