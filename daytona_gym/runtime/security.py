from __future__ import annotations

from collections.abc import Mapping

_SECRET_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "private_key",
)

DEFAULT_CAPTURE_LIMIT = 16_384


def clip_text(text: str, limit: int = DEFAULT_CAPTURE_LIMIT) -> tuple[str, bool]:
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]", True


def redact_env_vars(env: Mapping[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        lowered = key.lower().replace("-", "_")
        if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted
