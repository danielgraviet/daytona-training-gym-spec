from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
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
_API_KEY_ENV = "DAYTONA_API_KEY"
_API_KEY_FILE_ENV = "DAYTONA_API_KEY_FILE"


def clip_text(text: str, limit: int = DEFAULT_CAPTURE_LIMIT) -> tuple[str, bool]:
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]", True


def is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_FRAGMENTS)


def resolve_daytona_api_key(
    *,
    api_key: str | None = None,
    key_file: str | None = None,
) -> str | None:
    """Resolve the Daytona API key without requiring it in Ray runtime_env.

    Prefer an explicit key, then ``DAYTONA_API_KEY``, then the contents of
    ``DAYTONA_API_KEY_FILE`` (or ``key_file``). Ray ``job list`` dumps
    ``runtime_env`` literally — keep secrets on disk / in the shell, not in
    ``--runtime-env-json``.
    """
    if api_key is not None and str(api_key).strip():
        return str(api_key).strip()
    env_key = os.environ.get(_API_KEY_ENV)
    if env_key is not None and env_key.strip():
        return env_key.strip()
    path = key_file if key_file is not None else os.environ.get(_API_KEY_FILE_ENV)
    if not path:
        return None
    raw = Path(path).read_text(encoding="utf-8")
    return raw.strip() or None


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
