"""Duck-typed Daytona SDK for CPU CI only.

First-class proof of Daytona behavior lives in ``tests/e2e`` (real SDK + real
sandboxes). This fake must mirror shapes observed from live calls — see
module-level notes and AGENTS.md testing philosophy.

Observed live tool timeout (2026-09-21)::

    type: daytona.common.errors.DaytonaProcessExecutionTimeoutError
    str:  "Failed to execute command: command execution timeout"
    cause: ApiException 408 / code PROCESS_EXECUTION_TIMEOUT
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any


class FakeDaytonaProcessExecutionTimeoutError(Exception):
    """Mirrors ``daytona.common.errors.DaytonaProcessExecutionTimeoutError``."""


class FakeExecuteResponse:
    def __init__(self, exit_code: int = 0, result: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.result = result
        self.stderr = stderr
        self.artifacts = SimpleNamespace(stdout=result)


class FakeSandboxFs:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.uploads: list[tuple[str, bytes]] = []
        self.folders: list[str] = []

    async def upload_file(self, file: bytes, remote_path: str, timeout: int = 1800) -> None:
        del timeout
        payload = file if isinstance(file, bytes) else str(file).encode("utf-8")
        self.files[remote_path] = payload
        self.uploads.append((remote_path, payload))

    async def download_file(self, remote_path: str, timeout: int = 1800) -> bytes:
        del timeout
        if remote_path not in self.files:
            raise FileNotFoundError(f"{remote_path}: No such file")
        return self.files[remote_path]

    async def create_folder(self, path: str, mode: str, request_timeout: float | None = None) -> None:
        del mode, request_timeout
        self.folders.append(path)


class FakeSandboxProcess:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox
        self.commands: list[dict[str, Any]] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> FakeExecuteResponse:
        self.commands.append({"command": command, "cwd": cwd, "env": env, "timeout": timeout})
        if self._sandbox.hang_exec:
            # Same type-name + message shape as the live SDK timeout path.
            raise FakeDaytonaProcessExecutionTimeoutError(
                "Failed to execute command: command execution timeout"
            )
        if self._sandbox.fail_exec:
            raise RuntimeError("Failed to execute command: boom")
        if command.strip() == "false":
            return FakeExecuteResponse(exit_code=1, result="", stderr="false\n")
        if command.strip().startswith("echo "):
            return FakeExecuteResponse(result=command.strip()[5:] + "\n")
        if command.strip().startswith("cat "):
            path = command.strip()[4:].strip()
            if path not in self._sandbox.files:
                return FakeExecuteResponse(exit_code=1, stderr=f"cat: {path}: No such file\n")
            return FakeExecuteResponse(result=self._sandbox.files[path].decode("utf-8"))
        return FakeExecuteResponse(result="")


class FakeSandbox:
    def __init__(
        self,
        sandbox_id: str,
        *,
        hang_exec: bool = False,
        fail_exec: bool = False,
    ) -> None:
        self.id = sandbox_id
        self.files: dict[str, bytes] = {}
        self.hang_exec = hang_exec
        self.fail_exec = fail_exec
        self.fs = FakeSandboxFs(self.files)
        self.process = FakeSandboxProcess(self)
        self.deleted = False


class FakeAsyncDaytona:
    """Second-class stand-in for ``daytona.AsyncDaytona`` — CI only."""

    def __init__(
        self,
        *,
        fail_create: bool = False,
        fail_create_timeout: bool = False,
        hang_exec: bool = False,
        fail_exec: bool = False,
    ) -> None:
        self.fail_create = fail_create
        self.fail_create_timeout = fail_create_timeout
        self.hang_exec = hang_exec
        self.fail_exec = fail_exec
        self.created: list[Any] = []
        self.deleted: list[str] = []
        self.closed = False
        self._n = 0

    async def create(self, params: Any = None, *, timeout: float = 60) -> FakeSandbox:
        del timeout
        if self.fail_create_timeout:
            raise FakeDaytonaProcessExecutionTimeoutError("sandbox create timed out")
        if self.fail_create:
            raise RuntimeError("api rejected sandbox create")
        self._n += 1
        sandbox = FakeSandbox(
            f"sbx_{self._n}",
            hang_exec=self.hang_exec,
            fail_exec=self.fail_exec,
        )
        self.created.append({"sandbox": sandbox, "params": params})
        return sandbox

    async def delete(self, sandbox: FakeSandbox, timeout: float = 60, wait: bool = False) -> None:
        del timeout, wait
        sandbox.deleted = True
        self.deleted.append(sandbox.id)

    async def close(self) -> None:
        self.closed = True
