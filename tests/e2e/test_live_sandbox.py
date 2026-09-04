from __future__ import annotations

import pytest

from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.types import EnvironmentSpec, ToolAction, ToolName


@pytest.mark.e2e
async def test_live_sandbox_exec_write_read_and_cleanup() -> None:
    runtime = DaytonaEnvironmentRuntime(ephemeral=True, create_timeout_seconds=120)
    env = None
    try:
        env = await runtime.create(
            EnvironmentSpec(
                metadata={
                    "run_id": "e2e",
                    "rollout_id": "live-sandbox",
                    "project_id": "daytona-gym",
                }
            )
        )
        assert env.sandbox_id

        echoed = await runtime.execute(
            env,
            ToolAction(
                name=ToolName.RUN_COMMAND,
                arguments={"command": "echo daytona-gym-e2e"},
                timeout_seconds=30,
            ),
        )
        assert echoed.ok is True
        assert "daytona-gym-e2e" in echoed.stdout

        written = await runtime.execute(
            env,
            ToolAction(
                name=ToolName.WRITE_FILE,
                arguments={"path": "hello.txt", "content": "hello from e2e\n"},
                timeout_seconds=30,
            ),
        )
        assert written.ok is True

        read = await runtime.execute(
            env,
            ToolAction(
                name=ToolName.READ_FILE,
                arguments={"path": "hello.txt"},
                timeout_seconds=30,
            ),
        )
        assert read.ok is True
        assert "hello from e2e" in read.stdout
    finally:
        if env is not None:
            await runtime.close(env)
        await runtime.aclose()
        assert runtime.leaked_sandbox_ids == ()
