from __future__ import annotations

from typing import Protocol

from daytona_gym.runtime.types import EnvironmentHandle, EnvironmentSpec, ToolAction, ToolResult


class EnvironmentRuntime(Protocol):
    """Framework-neutral sandbox lifecycle. Must not accept trainer-native types."""

    async def create(self, spec: EnvironmentSpec) -> EnvironmentHandle: ...

    async def execute(self, env: EnvironmentHandle, action: ToolAction) -> ToolResult: ...

    async def reset(self, env: EnvironmentHandle) -> None: ...

    async def close(self, env: EnvironmentHandle) -> None: ...
