from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    SANDBOX_PROVISION_FAILED = "sandbox_provision_failed"
    SANDBOX_TIMEOUT = "sandbox_timeout"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_FAILED = "tool_failed"
    ROLLOUT_TIMEOUT = "rollout_timeout"
    INFERENCE_FAILED = "inference_failed"
    REWARD_FAILED = "reward_failed"
    USER_CODE_ERROR = "user_code_error"
    PLATFORM_ERROR = "platform_error"


_ABORTED_CODES = frozenset(
    {
        ErrorCode.SANDBOX_TIMEOUT,
        ErrorCode.TOOL_TIMEOUT,
        ErrorCode.ROLLOUT_TIMEOUT,
    }
)


class DaytonaError(Exception):
    """Structured failure for expected rollout/environment error classes."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}

    @property
    def rollout_status(self) -> str:
        """Slime-compatible status string for this error."""
        return "aborted" if self.code in _ABORTED_CODES else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "message": self.message,
            "details": self.details,
        }
