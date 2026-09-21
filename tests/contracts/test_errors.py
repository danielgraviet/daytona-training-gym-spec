from __future__ import annotations

from pathlib import Path

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.security import clip_text, redact_env_vars, sanitize_attributes
from daytona_gym.runtime.types import EnvironmentSpec, ToolAction, ToolName, ToolResult


def test_error_codes_match_spec() -> None:
    expected = {
        "sandbox_provision_failed",
        "sandbox_timeout",
        "tool_timeout",
        "tool_failed",
        "rollout_timeout",
        "inference_failed",
        "reward_failed",
        "user_code_error",
        "platform_error",
    }
    assert {str(code) for code in ErrorCode} == expected


def test_timeout_errors_map_to_aborted() -> None:
    assert DaytonaError(ErrorCode.ROLLOUT_TIMEOUT, "t").rollout_status == "aborted"
    assert DaytonaError(ErrorCode.TOOL_TIMEOUT, "t").rollout_status == "aborted"
    assert DaytonaError(ErrorCode.SANDBOX_PROVISION_FAILED, "t").rollout_status == "failed"


def test_looks_like_timeout_covers_sdk_shapes() -> None:
    from daytona_gym.runtime.daytona import _is_tool_timeout, _looks_like_timeout

    assert _looks_like_timeout(TimeoutError("timed out"))
    assert _looks_like_timeout(RuntimeError("Deadline exceeded"))
    assert _looks_like_timeout(RuntimeError("execution timeout after 5s"))
    assert not _looks_like_timeout(RuntimeError("command execution failed"))
    assert _is_tool_timeout(
        RuntimeError("command execution failed"),
        budget=5.0,
        elapsed=5.3,
    )
    assert not _is_tool_timeout(
        RuntimeError("command execution failed"),
        budget=5.0,
        elapsed=0.2,
    )


def test_redact_env_vars_hides_secrets() -> None:
    redacted = redact_env_vars(
        {
            "PATH": "/usr/bin",
            "DAYTONA_API_KEY": "super-secret",
            "TOKEN": "abc",
            "HOME": "/home/user",
        }
    )
    assert redacted["PATH"] == "/usr/bin"
    assert redacted["HOME"] == "/home/user"
    assert redacted["DAYTONA_API_KEY"] == "***"
    assert redacted["TOKEN"] == "***"


def test_resolve_daytona_api_key_from_file(tmp_path: Path, monkeypatch) -> None:
    from daytona_gym.runtime.security import resolve_daytona_api_key

    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY_FILE", raising=False)
    key_path = tmp_path / "daytona.key"
    key_path.write_text("file-secret\n", encoding="utf-8")
    assert resolve_daytona_api_key(key_file=str(key_path)) == "file-secret"
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(key_path))
    assert resolve_daytona_api_key() == "file-secret"
    monkeypatch.setenv("DAYTONA_API_KEY", "env-wins")
    assert resolve_daytona_api_key() == "env-wins"


def test_sanitize_attributes_redacts_secrets_and_stringifies_objects() -> None:
    sanitized = sanitize_attributes(
        {
            "run_id": "run_1",
            "api_key": "super-secret",
            "ok": True,
            "exit_code": 0,
            "nested": {"ignored": True},
        }
    )
    assert sanitized["run_id"] == "run_1"
    assert sanitized["api_key"] == "***"
    assert sanitized["ok"] is True
    assert sanitized["exit_code"] == 0
    assert sanitized["nested"] == "{'ignored': True}"


def test_clip_text_marks_truncation() -> None:
    text, truncated = clip_text("abcdef", limit=3)
    assert truncated is True
    assert text.startswith("abc")
    assert "...[truncated]" in text


async def test_tests_passed_reward_helper() -> None:
    from daytona_gym.runtime.reward import tests_passed_reward as tests_pass_reward

    runtime = FakeEnvironmentRuntime()
    env = await runtime.create(EnvironmentSpec())
    await runtime.execute(
        env,
        ToolAction(name=ToolName.WRITE_FILE, arguments={"path": "__tests_pass", "content": "1"}),
    )
    assert await tests_pass_reward(runtime, env) == 1.0
    await runtime.reset(env)
    runtime.script_command(
        env.sandbox_id,
        "pytest",
        ToolResult(ok=False, exit_code=1, stdout="", stderr="fail", duration_seconds=0.0),
    )
    assert await tests_pass_reward(runtime, env) == 0.0
    await runtime.close(env)
