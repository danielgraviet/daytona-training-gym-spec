from __future__ import annotations

import pytest

from daytona_gym.runtime.actions import parse_agent_action
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import ToolName


def test_parse_explicit_tool_and_final() -> None:
    tool = parse_agent_action(
        '{"type":"tool","name":"run_tests","arguments":{"command":"python test_broken.py"}}'
    )
    assert tool.is_final is False
    assert tool.parse_kind == "tool"
    assert tool.tool is not None
    assert tool.tool.name is ToolName.RUN_TESTS

    final = parse_agent_action('{"type":"final","content":"fixed"}')
    assert final.is_final is True
    assert final.parse_kind == "final"
    assert final.content == "fixed"
    assert final.coerced_from_non_json is False


def test_parse_markdown_fenced_tool_json() -> None:
    text = """```json
{"type":"tool","name":"write_file","arguments":{"path":"broken.py","content":"x"}}
```"""
    action = parse_agent_action(text)
    assert action.is_final is False
    assert action.parse_kind == "tool"
    assert action.tool is not None
    assert action.tool.name is ToolName.WRITE_FILE


def test_parse_trailing_junk_after_valid_tool_json() -> None:
    # Without raw_decode this used to coerce the whole string to final.
    text = (
        '{"type":"tool","name":"run_tests","arguments":{"command":"python test_broken.py"}}\n'
        "Sure, I ran the tests next I will fix the file..."
    )
    action = parse_agent_action(text)
    assert action.is_final is False
    assert action.parse_kind == "tool"
    assert action.tool is not None
    assert action.tool.name is ToolName.RUN_TESTS


def test_parse_prose_prefix_then_json_object() -> None:
    text = (
        "I will run the tests now:\n"
        '{"type":"tool","name":"run_tests","arguments":{"command":"python test_broken.py"}}'
    )
    action = parse_agent_action(text)
    assert action.is_final is False
    assert action.tool is not None
    assert action.tool.name is ToolName.RUN_TESTS


def test_parse_true_prose_still_final_fallback() -> None:
    action = parse_agent_action("The add function should return a + b instead of a - b.")
    assert action.is_final is True
    assert action.parse_kind == "final_fallback"
    assert action.coerced_from_non_json is True


def test_parse_unknown_tool_raises() -> None:
    with pytest.raises(DaytonaError) as caught:
        parse_agent_action('{"type":"tool","name":"explode","arguments":{}}')
    assert caught.value.code is ErrorCode.USER_CODE_ERROR
