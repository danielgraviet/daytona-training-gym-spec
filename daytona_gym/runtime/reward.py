from __future__ import annotations

from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.types import EnvironmentHandle, ToolAction, ToolName


async def tests_passed_reward(
    runtime: EnvironmentRuntime,
    env: EnvironmentHandle,
    test_command: str = "pytest",
    *,
    timeout_seconds: float | None = None,
) -> float:
    """User-owned helper: 1.0 if tests exit 0, else 0.0. Not used by core generate."""
    result = await runtime.execute(
        env,
        ToolAction(
            name=ToolName.RUN_TESTS,
            arguments={"command": test_command},
            timeout_seconds=timeout_seconds,
        ),
    )
    if result.ok and (result.exit_code or 0) == 0:
        return 1.0
    return 0.0
