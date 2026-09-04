from __future__ import annotations

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.security import clip_text, redact_env_vars
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
