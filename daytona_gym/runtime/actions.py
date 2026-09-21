from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import ToolAction, ToolName

_SPECIAL_TOKEN_RE = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|end\|>",
    re.IGNORECASE,
)
_TOOL_NAME_VALUES = frozenset(t.value for t in ToolName)
_DEFAULT_WRITE_PATH = "broken.py"
_DEFAULT_TEST_CMD = "python test_broken.py"


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

    Also accepted (models often emit these):
      {"type": "write_file", "arguments": {...}}   # type is the tool name
      {"type": "write_file", "content": "..."}      # path defaults to broken.py
      {"name": "run_tests", "arguments": {...}}    # omit type:"tool"
      multiple JSON objects in one turn — first *usable* action wins
    """
    stripped = text.strip()
    if not stripped:
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "empty model output")

    payloads = _extract_json_objects(stripped)
    if not payloads:
        return ParsedAction(
            is_final=True,
            content=stripped,
            parse_kind="final_fallback",
            coerced_from_non_json=True,
        )

    last_error: DaytonaError | None = None
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        try:
            action = _parse_payload(payload)
        except DaytonaError as exc:
            last_error = exc
            continue
        if action is not None:
            return action

    if last_error is not None:
        raise last_error
    raise DaytonaError(ErrorCode.USER_CODE_ERROR, "model output JSON must be an object")


def _parse_payload(payload: dict[str, Any]) -> ParsedAction | None:
    kind = payload.get("type")
    if kind == "final":
        content = payload.get("content", "")
        if not isinstance(content, str):
            raise DaytonaError(ErrorCode.USER_CODE_ERROR, "final content must be a string")
        return ParsedAction(is_final=True, content=content, parse_kind="final")

    if kind == "tool":
        return _tool_action(
            name=payload.get("name"),
            arguments=payload.get("arguments", {}),
            timeout=payload.get("timeout_seconds"),
            parse_kind="tool",
        )

    if isinstance(kind, str) and kind in _TOOL_NAME_VALUES:
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {
                k: v for k, v in payload.items() if k not in {"type", "timeout_seconds"}
            }
        return _tool_action(
            name=kind,
            arguments=arguments,
            timeout=payload.get("timeout_seconds"),
            parse_kind="tool_type_alias",
        )

    if kind is None and isinstance(payload.get("name"), str) and payload["name"] in _TOOL_NAME_VALUES:
        return _tool_action(
            name=payload.get("name"),
            arguments=payload.get("arguments", {}),
            timeout=payload.get("timeout_seconds"),
            parse_kind="tool_name_only",
        )

    return None


def _tool_action(
    *,
    name: Any,
    arguments: Any,
    timeout: Any,
    parse_kind: str,
) -> ParsedAction:
    try:
        tool_name = ToolName(name)
    except ValueError as exc:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"unknown tool name: {name!r}",
        ) from exc
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "tool arguments must be an object")
    if timeout is not None and not isinstance(timeout, (int, float)):
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, "timeout_seconds must be a number")
    normalized = _normalize_tool_arguments(tool_name, arguments)
    return ParsedAction(
        is_final=False,
        tool=ToolAction(
            name=tool_name,
            arguments=normalized,
            timeout_seconds=float(timeout) if timeout is not None else None,
        ),
        parse_kind=parse_kind,
    )


def _normalize_tool_arguments(tool_name: ToolName, arguments: dict[str, Any]) -> dict[str, Any]:
    """Fill coding-dogfood defaults when the model omits required fields."""
    args = dict(arguments)
    if tool_name is ToolName.WRITE_FILE:
        path = args.get("path")
        content = args.get("content")
        if (not isinstance(path, str) or not path.strip()) and isinstance(content, str):
            args["path"] = os.environ.get("DAYTONA_DEFAULT_WRITE_PATH", _DEFAULT_WRITE_PATH)
    if tool_name is ToolName.RUN_TESTS:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            args["command"] = os.environ.get(
                "DAYTONA_BOOTSTRAP_RUN_TESTS_CMD",
                _DEFAULT_TEST_CMD,
            )
    if tool_name is ToolName.READ_FILE:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            args["path"] = os.environ.get("DAYTONA_DEFAULT_WRITE_PATH", _DEFAULT_WRITE_PATH)
    return args


def _extract_json_objects(text: str) -> list[Any]:
    candidate = _strip_markdown_fence(text)
    objects: list[Any] = []
    try:
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            return [loaded]
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(candidate):
        start = candidate.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = end
    return objects


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    return stripped.strip()
