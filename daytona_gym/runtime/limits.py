from __future__ import annotations

import asyncio
from typing import Any

from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import EnvironmentHandle, EnvironmentSpec, ToolAction, ToolResult

_RETRYABLE = frozenset({ErrorCode.SANDBOX_PROVISION_FAILED, ErrorCode.PLATFORM_ERROR})


class RetryingEnvironmentRuntime:
    """Retry sandbox provision for transient platform failures."""

    def __init__(
        self,
        inner: EnvironmentRuntime,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 0.05,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    async def create(self, spec: EnvironmentSpec) -> EnvironmentHandle:
        last_error: DaytonaError | None = None
        for attempt in range(self._max_attempts):
            try:
                return await self._inner.create(spec)
            except DaytonaError as exc:
                if exc.code not in _RETRYABLE or attempt + 1 >= self._max_attempts:
                    raise
                last_error = exc
                delay = self._backoff_seconds * (2**attempt)
                if delay:
                    await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def execute(self, env: EnvironmentHandle, action: ToolAction) -> ToolResult:
        return await self._inner.execute(env, action)

    async def reset(self, env: EnvironmentHandle) -> None:
        await self._inner.reset(env)

    async def close(self, env: EnvironmentHandle) -> None:
        await self._inner.close(env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class LimitingEnvironmentRuntime:
    """Cap how many sandboxes may be in flight at once."""

    def __init__(self, inner: EnvironmentRuntime, max_in_flight: int) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self._inner = inner
        self._sem = asyncio.Semaphore(max_in_flight)
        self._held: set[str] = set()
        self.peak_in_flight = 0

    @property
    def in_flight(self) -> int:
        return len(self._held)

    async def create(self, spec: EnvironmentSpec) -> EnvironmentHandle:
        await self._sem.acquire()
        try:
            handle = await self._inner.create(spec)
        except BaseException:
            self._sem.release()
            raise
        self._held.add(handle.sandbox_id)
        self.peak_in_flight = max(self.peak_in_flight, len(self._held))
        return handle

    async def execute(self, env: EnvironmentHandle, action: ToolAction) -> ToolResult:
        return await self._inner.execute(env, action)

    async def reset(self, env: EnvironmentHandle) -> None:
        await self._inner.reset(env)

    async def close(self, env: EnvironmentHandle) -> None:
        try:
            await self._inner.close(env)
        finally:
            if env.sandbox_id in self._held:
                self._held.discard(env.sandbox_id)
                self._sem.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
