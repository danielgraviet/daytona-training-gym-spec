from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import ToolAction, ToolName


@dataclass(frozen=True)
class ParsedAction:
    is_final: bool
    content: str | None = None
    tool: ToolAction | None = None


def parse_agent_action(text: str) -> ParsedAction:
    """Parse one model turn into a tool call or a final answer.

    Expected JSON:
      {"type": "final", "content": "..."}
      {"type": "tool", "name": "run_command", "arguments": {"command": "echo hi"}}
    Non-JSON text is treated as a final answer so simple completions still work.
    """
    stripped = text.strip()
    if not stripped:
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "empty model output")

    try:
        payload: Any = json.loads(stripped)
    except json.JSONDecodeError:
        return ParsedAction(is_final=True, content=stripped)

    if not isinstance(payload, dict):
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "model output JSON must be an object")

    kind = payload.get("type")
    if kind == "final":
        content = payload.get("content", "")
        if not isinstance(content, str):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "final content must be a string")
        return ParsedAction(is_final=True, content=content)

    if kind == "tool":
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        timeout = payload.get("timeout_seconds")
        try:
            tool_name = ToolName(name)
        except ValueError as exc:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                f"unknown tool name: {name!r}",
            ) from exc
        if not isinstance(arguments, dict):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "tool arguments must be an object")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "timeout_seconds must be a number")
        return ParsedAction(
            is_final=False,
            tool=ToolAction(
                name=tool_name,
                arguments=arguments,
                timeout_seconds=float(timeout) if timeout is not None else None,
            ),
        )

    raise DaytonaError(ErrorCode.USER_CODE_ERROR, f"unknown action type: {kind!r}")
