from __future__ import annotations

import asyncio
import posixpath
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NoReturn

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.security import DEFAULT_CAPTURE_LIMIT, clip_text, resolve_daytona_api_key
from daytona_gym.runtime.types import (
    EnvironmentHandle,
    EnvironmentSpec,
    ToolAction,
    ToolName,
    ToolResult,
)

_LABEL_KEYS = ("run_id", "rollout_id", "project_id", "sample_id")
_BLOCKED_ENV_KEYS = frozenset(
    {
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "OPENAI_API_KEY",
    }
)
_PATCH_REMOTE_PATH = "/tmp/daytona-gym.patch"


@dataclass
class _SandboxRecord:
    sandbox: Any
    closed: bool = False


class DaytonaEnvironmentRuntime:
    """EnvironmentRuntime backed by the official async Daytona SDK."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        auto_stop_interval: int = 0,
        ephemeral: bool = True,
        stdout_limit: int = DEFAULT_CAPTURE_LIMIT,
        create_timeout_seconds: float = 60,
        env_vars: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._api_key = api_key
        self._api_url = api_url
        self._target = target
        self._auto_stop_interval = auto_stop_interval
        self._ephemeral = ephemeral
        self._stdout_limit = stdout_limit
        self._create_timeout_seconds = create_timeout_seconds
        self._env_vars = dict(env_vars or {})
        self._sandboxes: dict[str, _SandboxRecord] = {}
        self._lock = asyncio.Lock()
        self._sdk: Any | None = None

    @classmethod
    def from_args(cls, args: Any) -> DaytonaEnvironmentRuntime:
        timeout = getattr(args, "daytona_sandbox_timeout_seconds", None)
        api_key = getattr(args, "daytona_api_key", None)
        if not api_key:
            api_key = resolve_daytona_api_key()
        return cls(
            api_key=api_key,
            api_url=getattr(args, "daytona_api_url", None),
            target=getattr(args, "daytona_target", None),
            auto_stop_interval=int(getattr(args, "daytona_auto_stop_interval", 0) or 0),
            ephemeral=bool(getattr(args, "daytona_ephemeral", True)),
            stdout_limit=int(getattr(args, "daytona_stdout_limit", DEFAULT_CAPTURE_LIMIT)),
            create_timeout_seconds=float(timeout) if timeout is not None else 60.0,
            env_vars=getattr(args, "daytona_env_vars", None),
        )

    @property
    def leaked_sandbox_ids(self) -> tuple[str, ...]:
        return tuple(
            sandbox_id for sandbox_id, record in self._sandboxes.items() if not record.closed
        )

    async def create(self, spec: EnvironmentSpec) -> EnvironmentHandle:
        client = await self._ensure_client()
        params = self._params_for_spec(spec)
        timeout = (
            spec.timeout_seconds
            if spec.timeout_seconds is not None
            else self._create_timeout_seconds
        )
        sandbox: Any | None = None
        try:
            sandbox = await client.create(params, timeout=_sdk_timeout(timeout))
        except asyncio.CancelledError:
            raise
        except DaytonaError:
            raise
        except Exception as exc:
            _reraise_sdk(
                exc,
                timeout_code=ErrorCode.SANDBOX_TIMEOUT,
                fallback=ErrorCode.SANDBOX_PROVISION_FAILED,
                message="sandbox provision failed",
            )
        if sandbox is None or not getattr(sandbox, "id", None):
            raise DaytonaError(
                ErrorCode.SANDBOX_PROVISION_FAILED,
                "sandbox create returned no id",
            )

        handle = EnvironmentHandle(
            sandbox_id=str(sandbox.id),
            run_id=spec.metadata.get("run_id", ""),
            rollout_id=spec.metadata.get("rollout_id", ""),
        )
        async with self._lock:
            self._sandboxes[handle.sandbox_id] = _SandboxRecord(sandbox=sandbox)
        return handle

    async def execute(self, env: EnvironmentHandle, action: ToolAction) -> ToolResult:
        sandbox = await self._require_open(env)
        started = time.perf_counter()
        budget = action.timeout_seconds
        try:
            raw = await _await_bounded(self._dispatch(sandbox, action), budget)
        except asyncio.CancelledError:
            raise
        except DaytonaError:
            raise
        except TimeoutError as exc:
            raise DaytonaError(
                ErrorCode.TOOL_TIMEOUT,
                f"tool {action.name} timed out",
                details={"tool": str(action.name), "timeout_seconds": budget},
            ) from exc
        except Exception as exc:
            elapsed = time.perf_counter() - started
            if _is_tool_timeout(exc, budget=budget, elapsed=elapsed):
                raise DaytonaError(
                    ErrorCode.TOOL_TIMEOUT,
                    f"tool {action.name} timed out",
                    details={
                        "tool": str(action.name),
                        "timeout_seconds": budget,
                        "elapsed_seconds": elapsed,
                        "cause": type(exc).__name__,
                    },
                ) from exc
            _reraise_sdk(
                exc,
                timeout_code=ErrorCode.TOOL_TIMEOUT,
                fallback=ErrorCode.TOOL_FAILED,
                message=f"tool {action.name} failed",
            )
        duration = time.perf_counter() - started
        stdout, stdout_truncated = clip_text(raw.stdout, self._stdout_limit)
        stderr, stderr_truncated = clip_text(raw.stderr, self._stdout_limit)
        return ToolResult(
            ok=raw.ok,
            exit_code=raw.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            truncated=raw.truncated or stdout_truncated or stderr_truncated,
        )

    async def reset(self, env: EnvironmentHandle) -> None:
        sandbox = await self._require_open(env)
        try:
            await sandbox.process.exec(
                "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; "
                "then git reset --hard HEAD && git clean -fd; fi",
            )
        except asyncio.CancelledError:
            raise
        except DaytonaError:
            raise
        except Exception as exc:
            _reraise_sdk(
                exc,
                timeout_code=ErrorCode.TOOL_TIMEOUT,
                fallback=ErrorCode.PLATFORM_ERROR,
                message="sandbox reset failed",
            )

    async def close(self, env: EnvironmentHandle) -> None:
        async with self._lock:
            record = self._sandboxes.get(env.sandbox_id)
            if record is None or record.closed:
                return
            record.closed = True
            sandbox = record.sandbox
        client = self._client
        if client is None:
            return
        try:
            await client.delete(sandbox, wait=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def aclose(self) -> None:
        """Close the shared SDK HTTP client if this runtime created it."""
        client = self._client
        if client is None or not self._owns_client:
            return
        close = getattr(client, "close", None)
        if close is None:
            return
        await close()

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        sdk = self._load_sdk()
        config_cls = getattr(sdk, "DaytonaConfig", None)
        client_cls = getattr(sdk, "AsyncDaytona", None)
        if client_cls is None:
            raise DaytonaError(ErrorCode.PLATFORM_ERROR, "daytona.AsyncDaytona is unavailable")
        config = None
        if config_cls is not None:
            config_kwargs: dict[str, Any] = {"connection_pool_maxsize": None}
            if self._api_key:
                config_kwargs["api_key"] = self._api_key
            if self._api_url:
                config_kwargs["api_url"] = self._api_url
            if self._target:
                config_kwargs["target"] = self._target
            config = config_cls(**config_kwargs)
        self._client = client_cls(config) if config is not None else client_cls()
        self._owns_client = True
        return self._client

    def _load_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        try:
            import daytona as sdk
        except ImportError as exc:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                "the daytona SDK is not installed; pip install daytona",
            ) from exc
        self._sdk = sdk
        return sdk

    def _params_for_spec(self, spec: EnvironmentSpec) -> Any:
        labels = {"component": "daytona-gym"}
        for key in _LABEL_KEYS:
            value = spec.metadata.get(key)
            if value:
                labels[key] = value
        common: dict[str, Any] = {
            "language": "python",
            "labels": labels,
            "auto_stop_interval": self._auto_stop_interval,
            "ephemeral": self._ephemeral,
        }
        env_vars = _sandbox_env_vars(self._env_vars)
        if env_vars:
            common["env_vars"] = env_vars
        if spec.snapshot:
            return self._typed_params(
                "CreateSandboxFromSnapshotParams",
                snapshot=spec.snapshot,
                **common,
            )
        if spec.image:
            return self._typed_params("CreateSandboxFromImageParams", image=spec.image, **common)
        return self._typed_params("CreateSandboxFromSnapshotParams", **common)

    def _typed_params(self, class_name: str, **kwargs: Any) -> Any:
        if self._sdk is None:
            try:
                self._sdk = __import__("daytona")
            except ImportError:
                self._sdk = None
        cls = getattr(self._sdk, class_name, None) if self._sdk is not None else None
        if cls is not None:
            return cls(**kwargs)
        return SimpleNamespace(kind=class_name, **kwargs)

    async def _require_open(self, env: EnvironmentHandle) -> Any:
        async with self._lock:
            record = self._sandboxes.get(env.sandbox_id)
        if record is None:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                f"unknown sandbox: {env.sandbox_id}",
                details={"sandbox_id": env.sandbox_id},
            )
        if record.closed:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                f"sandbox already closed: {env.sandbox_id}",
                details={"sandbox_id": env.sandbox_id},
            )
        return record.sandbox

    async def _dispatch(self, sandbox: Any, action: ToolAction) -> ToolResult:
        args = dict(action.arguments)
        timeout = _exec_timeout(action.timeout_seconds)
        if action.name == ToolName.RUN_COMMAND:
            return await self._run_command(sandbox, args, timeout)
        if action.name == ToolName.READ_FILE:
            return await self._read_file(sandbox, args, timeout)
        if action.name == ToolName.WRITE_FILE:
            return await self._write_file(sandbox, args, timeout)
        if action.name == ToolName.APPLY_PATCH:
            return await self._apply_patch(sandbox, args, timeout)
        if action.name == ToolName.RUN_TESTS:
            return await self._run_tests(sandbox, args, timeout)
        raise DaytonaError(ErrorCode.TOOL_FAILED, f"unsupported tool: {action.name}")

    async def _run_command(self, sandbox: Any, args: dict[str, Any], timeout: int | None) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "run_command requires a command string")
        cwd = args.get("cwd")
        env = args.get("env")
        if env is not None and not isinstance(env, Mapping):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "run_command env must be a mapping")
        response = await sandbox.process.exec(
            command.strip(),
            cwd=cwd if isinstance(cwd, str) else None,
            env=_sandbox_env_vars(env) if env is not None else None,
            timeout=timeout,
        )
        return _from_exec_response(response)

    async def _read_file(self, sandbox: Any, args: dict[str, Any], timeout: int | None) -> ToolResult:
        path = _require_path(args)
        try:
            if timeout is None:
                content = await sandbox.fs.download_file(path)
            else:
                content = await sandbox.fs.download_file(path, int(timeout))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _looks_missing(exc):
                return ToolResult(
                    ok=False,
                    exit_code=1,
                    stdout="",
                    stderr=f"read_file: {path}: No such file\n",
                    duration_seconds=0.0,
                )
            raise
        text = _bytes_to_text(content)
        return ToolResult(ok=True, exit_code=0, stdout=text, stderr="", duration_seconds=0.0)

    async def _write_file(self, sandbox: Any, args: dict[str, Any], timeout: int | None) -> ToolResult:
        path = _require_path(args)
        content = args.get("content")
        if not isinstance(content, str):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "write_file requires string content")
        parent = posixpath.dirname(path)
        if parent and parent not in {".", "/"}:
            try:
                await sandbox.fs.create_folder(parent, "755")
            except Exception:
                pass
        payload = content.encode("utf-8")
        await sandbox.fs.upload_file(payload, path, timeout=timeout or 30 * 60)
        return ToolResult(ok=True, exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    async def _apply_patch(self, sandbox: Any, args: dict[str, Any], timeout: int | None) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        if isinstance(path, str) and isinstance(content, str):
            return await self._write_file(sandbox, {"path": path, "content": content}, timeout)
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "apply_patch requires path+content or patch",
            )
        await sandbox.fs.upload_file(
            patch.encode("utf-8"),
            _PATCH_REMOTE_PATH,
            timeout=timeout or 30 * 60,
        )
        response = await sandbox.process.exec(
            f"git apply --whitespace=nowarn {_PATCH_REMOTE_PATH}",
            timeout=timeout,
        )
        return _from_exec_response(response)

    async def _run_tests(self, sandbox: Any, args: dict[str, Any], timeout: int | None) -> ToolResult:
        command = args.get("command", "pytest")
        if not isinstance(command, str) or not command.strip():
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "run_tests command must be a string")
        response = await sandbox.process.exec(command.strip(), timeout=timeout)
        return _from_exec_response(response)


def _sandbox_env_vars(env: Mapping[str, str] | None) -> dict[str, str] | None:
    if env is None:
        return None
    cleaned: dict[str, str] = {}
    for key, value in env.items():
        if str(key).upper() in _BLOCKED_ENV_KEYS:
            continue
        cleaned[str(key)] = str(value)
    return cleaned


def _require_path(args: dict[str, Any]) -> str:
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "path is required")
    path = path.strip()
    parts = posixpath.normpath(path).split("/")
    if ".." in parts:
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "path must not contain '..'")
    return path


def _bytes_to_text(content: Any) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _from_exec_response(response: Any) -> ToolResult:
    exit_code = int(getattr(response, "exit_code", 1) or 0)
    stdout = getattr(response, "result", None)
    if stdout is None:
        artifacts = getattr(response, "artifacts", None)
        stdout = getattr(artifacts, "stdout", "") if artifacts is not None else ""
    stderr = getattr(response, "stderr", "") or ""
    return ToolResult(
        ok=exit_code == 0,
        exit_code=exit_code,
        stdout=str(stdout or ""),
        stderr=str(stderr),
        duration_seconds=0.0,
    )


def _looks_like_timeout(exc: BaseException) -> bool:
    """Heuristic: Daytona/HTTP clients often raise non-TimeoutError on stall."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name or "timedout" in name:
        return True
    text = str(exc).lower()
    if "unexpected keyword argument" in text:
        return False
    needles = (
        "timed out",
        "timeout exceeded",
        "operation timed out",
        "deadline exceeded",
        "timeout waiting",
        "request timeout",
        "read timed out",
        "gateway timeout",
        "context deadline",
        "execution timeout",
        "command timeout",
        "timeout of",
        "time out",
    )
    return any(needle in text for needle in needles)


def _near_timeout_budget(budget: float | None, elapsed: float) -> bool:
    """True when wall clock is close to the configured tool timeout.

    Edge dogfood: Daytona killed a 5s-budget sleep after ~5.3s but raised a
    generic error → previously labeled tool_failed. Treat near-budget failures
    as timeouts when a budget was set.
    """
    if budget is None or budget <= 0:
        return False
    return elapsed >= float(budget) * 0.9


def _is_tool_timeout(
    exc: BaseException,
    *,
    budget: float | None,
    elapsed: float,
) -> bool:
    return _looks_like_timeout(exc) or _near_timeout_budget(budget, elapsed)


async def _await_bounded(coro: Any, budget: float | None) -> Any:
    """Client-side backstop when the SDK ignores / mis-reports timeouts."""
    if budget is None or budget <= 0:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=float(budget))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"timed out after {budget}s") from exc


def _looks_missing(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "not found" in text
        or "no such file" in text
        or type(exc).__name__ in {"FileNotFoundError", "NotFoundException"}
    )


def _sdk_timeout(seconds: float | None) -> float:
    if seconds is None:
        return 60.0
    return float(seconds)


def _exec_timeout(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    if seconds <= 0:
        return None
    return max(1, int(seconds))


def _reraise_sdk(
    exc: BaseException,
    *,
    timeout_code: ErrorCode,
    fallback: ErrorCode,
    message: str,
) -> NoReturn:
    timeout_like = _looks_like_timeout(exc)
    code = timeout_code if timeout_like else fallback
    raise DaytonaError(code, message, details={"cause": type(exc).__name__}) from exc
