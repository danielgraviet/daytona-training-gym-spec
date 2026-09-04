from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.security import DEFAULT_CAPTURE_LIMIT, clip_text
from daytona_gym.runtime.types import (
    EnvironmentHandle,
    EnvironmentSpec,
    ToolAction,
    ToolName,
    ToolResult,
)
from daytona_gym.telemetry.ids import new_sandbox_id


@dataclass
class _SandboxState:
    handle: EnvironmentHandle
    files: dict[str, str] = field(default_factory=dict)
    closed: bool = False
    reset_count: int = 0
    scripted_commands: dict[str, ToolResult] = field(default_factory=dict)


class FakeEnvironmentRuntime:
    """In-memory coding environment used for CPU-only contract tests."""

    def __init__(
        self,
        *,
        fail_create: bool = False,
        create_delay_seconds: float = 0.0,
        tool_delay_seconds: float = 0.0,
        hang_on_create: bool = False,
        hang_on_execute: bool = False,
        stdout_limit: int = DEFAULT_CAPTURE_LIMIT,
    ) -> None:
        self._fail_create = fail_create
        self._create_delay_seconds = create_delay_seconds
        self._tool_delay_seconds = tool_delay_seconds
        self._hang_on_create = hang_on_create
        self._hang_on_execute = hang_on_execute
        self._stdout_limit = stdout_limit
        self._hang = asyncio.Event()
        self._sandboxes: dict[str, _SandboxState] = {}
        self._created_ids: list[str] = []
        self._closed_ids: list[str] = []

    def release_hang(self) -> None:
        self._hang.set()

    def script_command(self, sandbox_id: str, command: str, result: ToolResult) -> None:
        state = self._require(sandbox_id)
        state.scripted_commands[command] = result

    @property
    def created_ids(self) -> tuple[str, ...]:
        return tuple(self._created_ids)

    @property
    def closed_ids(self) -> tuple[str, ...]:
        return tuple(self._closed_ids)

    @property
    def leaked_sandbox_ids(self) -> tuple[str, ...]:
        return tuple(
            sandbox_id
            for sandbox_id, state in self._sandboxes.items()
            if not state.closed
        )

    def files_for(self, sandbox_id: str) -> dict[str, str]:
        return dict(self._require(sandbox_id).files)

    async def create(self, spec: EnvironmentSpec) -> EnvironmentHandle:
        if self._fail_create or spec.metadata.get("fake.fail_create") == "true":
            raise DaytonaError(
                ErrorCode.SANDBOX_PROVISION_FAILED,
                "fake sandbox provision failed",
                details={"image": spec.image},
            )
        if self._create_delay_seconds:
            await asyncio.sleep(self._create_delay_seconds)
        if self._hang_on_create:
            await self._hang.wait()

        sandbox_id = new_sandbox_id()
        handle = EnvironmentHandle(
            sandbox_id=sandbox_id,
            run_id=spec.metadata.get("run_id", ""),
            rollout_id=spec.metadata.get("rollout_id", ""),
        )
        self._sandboxes[sandbox_id] = _SandboxState(handle=handle)
        self._created_ids.append(sandbox_id)
        return handle

    async def execute(self, env: EnvironmentHandle, action: ToolAction) -> ToolResult:
        state = self._require(env.sandbox_id, allow_closed=False)
        started = time.perf_counter()
        timeout = action.timeout_seconds

        async def _run() -> ToolResult:
            if self._tool_delay_seconds:
                await asyncio.sleep(self._tool_delay_seconds)
            if self._hang_on_execute:
                await self._hang.wait()
            return await self._dispatch(state, action)

        try:
            if timeout is not None:
                result = await asyncio.wait_for(_run(), timeout=timeout)
            else:
                result = await _run()
        except TimeoutError as exc:
            raise DaytonaError(
                ErrorCode.TOOL_TIMEOUT,
                f"tool {action.name} timed out",
                details={"tool": str(action.name), "timeout_seconds": timeout},
            ) from exc

        duration = time.perf_counter() - started
        stdout, stdout_truncated = clip_text(result.stdout, self._stdout_limit)
        stderr, stderr_truncated = clip_text(result.stderr, self._stdout_limit)
        return ToolResult(
            ok=result.ok,
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            truncated=result.truncated or stdout_truncated or stderr_truncated,
        )

    async def reset(self, env: EnvironmentHandle) -> None:
        state = self._require(env.sandbox_id, allow_closed=False)
        state.files.clear()
        state.scripted_commands.clear()
        state.reset_count += 1

    async def close(self, env: EnvironmentHandle) -> None:
        state = self._sandboxes.get(env.sandbox_id)
        if state is None or state.closed:
            return
        state.closed = True
        self._closed_ids.append(env.sandbox_id)

    async def _dispatch(self, state: _SandboxState, action: ToolAction) -> ToolResult:
        args = dict(action.arguments)
        if action.name == ToolName.RUN_COMMAND:
            return self._run_command(state, args)
        if action.name == ToolName.READ_FILE:
            return self._read_file(state, args)
        if action.name == ToolName.WRITE_FILE:
            return self._write_file(state, args)
        if action.name == ToolName.APPLY_PATCH:
            return self._apply_patch(state, args)
        if action.name == ToolName.RUN_TESTS:
            return self._run_tests(state, args)
        raise DaytonaError(ErrorCode.TOOL_FAILED, f"unsupported tool: {action.name}")

    def _run_command(self, state: _SandboxState, args: dict[str, Any]) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "run_command requires a command string")
        command = command.strip()
        if command in state.scripted_commands:
            return state.scripted_commands[command]
        return self._builtin_command(state, command)

    def _builtin_command(self, state: _SandboxState, command: str) -> ToolResult:
        if command == "true":
            return _ok("")
        if command == "false":
            return _fail("false", exit_code=1)
        if command.startswith("echo "):
            return _ok(command[5:] + "\n")
        if command.startswith("sleep "):
            # Delay is handled by the caller via tool_delay/timeout; treat as success.
            return _ok("")
        if command.startswith("cat "):
            path = command[4:].strip()
            if path not in state.files:
                return _fail(f"cat: {path}: No such file\n", exit_code=1)
            return _ok(state.files[path])
        if command == "ls":
            listing = "\n".join(sorted(state.files))
            return _ok(listing + ("\n" if listing else ""))
        return _ok("")

    def _read_file(self, state: _SandboxState, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "read_file requires path")
        if path not in state.files:
            return _fail(f"read_file: {path}: No such file\n", exit_code=1)
        return _ok(state.files[path])

    def _write_file(self, state: _SandboxState, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "write_file requires path")
        if not isinstance(content, str):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "write_file requires string content")
        state.files[path] = content
        return _ok("")

    def _apply_patch(self, state: _SandboxState, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        if isinstance(path, str) and isinstance(content, str):
            state.files[path] = content
            return _ok("")
        patch = args.get("patch")
        if isinstance(patch, str) and patch:
            state.files.setdefault("_last.patch", patch)
            return _ok("")
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "apply_patch requires path+content or patch",
        )

    def _run_tests(self, state: _SandboxState, args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "pytest")
        if not isinstance(command, str):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "run_tests command must be a string")
        if command in state.scripted_commands:
            return state.scripted_commands[command]
        if state.files.get("__tests_pass") == "1":
            return _ok("tests passed\n")
        return _fail("tests failed\n", exit_code=1)

    def _require(self, sandbox_id: str, *, allow_closed: bool = True) -> _SandboxState:
        state = self._sandboxes.get(sandbox_id)
        if state is None:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                f"unknown sandbox: {sandbox_id}",
                details={"sandbox_id": sandbox_id},
            )
        if state.closed and not allow_closed:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                f"sandbox already closed: {sandbox_id}",
                details={"sandbox_id": sandbox_id},
            )
        return state


def _ok(stdout: str, *, exit_code: int = 0) -> ToolResult:
    return ToolResult(
        ok=exit_code == 0,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_seconds=0.0,
    )


def _fail(stderr: str, *, exit_code: int = 1) -> ToolResult:
    return ToolResult(
        ok=False,
        exit_code=exit_code,
        stdout="",
        stderr=stderr,
        duration_seconds=0.0,
    )
