from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import ToolAction, ToolName

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|end\|>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    is_final: bool
    content: str | None = None
    tool: ToolAction | None = None
    # How the turn was interpreted — useful for harness debugging.
    parse_kind: str = "final"
    # True when free text / junk was coerced to final (not explicit {"type":"final"}).
    coerced_from_non_json: bool = False


def sanitize_generation_text(text: str) -> str:
    """Strip chat special tokens before appending model text to the conversation."""
    cleaned = _SPECIAL_TOKEN_RE.sub("", text)
    return cleaned.strip()


def parse_agent_action(text: str) -> ParsedAction:
    """Parse one model turn into a tool call or a final answer.

    Expected JSON:
      {"type": "final", "content": "..."}
      {"type": "tool", "name": "run_tests", "arguments": {"command": "..."}}

    Resilience (matches what stronger models often emit):
      - markdown fences around JSON are stripped
      - first JSON object is taken via raw_decode (trailing junk after a valid
        object no longer forces a ``final`` — Slime Search-R1 hit the same class
        of bug without stop strings / postprocess)
      - true non-JSON prose still becomes final so simple completions work
    """
    stripped = text.strip()
    if not stripped:
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "empty model output")

    payload = _extract_json_object(stripped)
    if payload is None:
        return ParsedAction(
            is_final=True,
            content=stripped,
            parse_kind="final_fallback",
            coerced_from_non_json=True,
        )

    if not isinstance(payload, dict):
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "model output JSON must be an object")

    kind = payload.get("type")
    if kind == "final":
        content = payload.get("content", "")
        if not isinstance(content, str):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "final content must be a string")
        return ParsedAction(is_final=True, content=content, parse_kind="final")

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
            parse_kind="tool",
        )

    raise DaytonaError(ErrorCode.USER_CODE_ERROR, f"unknown action type: {kind!r}")


def _extract_json_object(text: str) -> Any | None:
    candidate = _strip_markdown_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(candidate, start)
    except json.JSONDecodeError:
        return None
    return obj


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    # Drop opening ``` / ```json and a trailing fence if present.
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    return stripped.strip()
