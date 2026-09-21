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


def test_sanitize_strips_im_end() -> None:
    from daytona_gym.runtime.actions import sanitize_generation_text

    assert (
        sanitize_generation_text(
            '{"type":"final","content":"fixed"}<|im_end|>'
        )
        == '{"type":"final","content":"fixed"}'
    )


def test_parse_unknown_tool_raises() -> None:
    with pytest.raises(DaytonaError) as caught:
        parse_agent_action('{"type":"tool","name":"explode","arguments":{}}')
    assert caught.value.code is ErrorCode.USER_CODE_ERROR


def test_parse_type_is_tool_name_alias() -> None:
    action = parse_agent_action(
        '{"type":"write_file","arguments":{"path":"broken.py","content":"x"}}'
    )
    assert action.is_final is False
    assert action.parse_kind == "tool_type_alias"
    assert action.tool is not None
    assert action.tool.name is ToolName.WRITE_FILE
    assert action.tool.arguments["path"] == "broken.py"


def test_parse_write_file_content_only_defaults_path() -> None:
    action = parse_agent_action(
        '{"type":"write_file","content":"def add(a, b):\\n    return a + b\\n"}'
    )
    assert action.tool is not None
    assert action.tool.name is ToolName.WRITE_FILE
    assert action.tool.arguments["path"] == "broken.py"
    assert "a + b" in action.tool.arguments["content"]


def test_parse_run_tests_without_command_defaults() -> None:
    action = parse_agent_action('{"type":"run_tests"}')
    assert action.tool is not None
    assert action.tool.name is ToolName.RUN_TESTS
    assert action.tool.arguments["command"] == "python test_broken.py"


def test_parse_picks_usable_json_among_echoed_tool_result() -> None:
    text = (
        '<tool_result name="read_file" ok="true" exit_code="0" truncated="false">\n'
        "def add(a, b):\n    return a - b\n"
        "</tool_result> "
        '{"type":"write_file","content":"def add(a, b):\\n    return a + b\\n"} '
        '{"type":"run_tests"} '
        '{"type":"final","content":"fixed"}'
    )
    action = parse_agent_action(text)
    assert action.is_final is False
    assert action.tool is not None
    assert action.tool.name is ToolName.WRITE_FILE
    assert action.tool.arguments["path"] == "broken.py"


def test_parse_agent_actions_returns_write_tests_final_in_order() -> None:
    from daytona_gym.runtime.actions import parse_agent_actions

    text = (
        '{"type":"write_file","content":"def add(a, b):\\n    return a + b\\n"} '
        '{"type":"run_tests"} '
        '{"type":"final","content":"fixed"}'
    )
    actions = parse_agent_actions(text)
    assert [a.is_final for a in actions] == [False, False, True]
    assert actions[0].tool is not None and actions[0].tool.name is ToolName.WRITE_FILE
    assert actions[1].tool is not None and actions[1].tool.name is ToolName.RUN_TESTS
    assert actions[2].content == "fixed"


def test_hallucinated_tool_result_detected() -> None:
    from daytona_gym.runtime.actions import looks_like_hallucinated_tool_result

    assert looks_like_hallucinated_tool_result(
        '<tool_result name="run_tests" ok="true" exit_code="0" truncated="false">OK</tool_result>'
    )
    assert not looks_like_hallucinated_tool_result(
        '{"type":"run_tests","arguments":{"command":"python test_broken.py"}}'
    )


def test_parse_name_only_tool() -> None:
    action = parse_agent_action(
        '{"name":"run_tests","arguments":{"command":"python test_broken.py"}}'
    )
    assert action.is_final is False
    assert action.parse_kind == "tool_name_only"
    assert action.tool is not None
    assert action.tool.name is ToolName.RUN_TESTS
