from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_FRAGMENTS)


def redact_env_vars(env: Mapping[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        if is_secret_key(key):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def sanitize_attributes(attributes: Mapping[str, object]) -> dict[str, Any]:
    """Make span attributes JSON-safe and redact secret-looking keys."""
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        name = str(key)
        if is_secret_key(name):
            sanitized[name] = "***"
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[name] = value
        else:
            sanitized[name] = str(value)
    return sanitized
