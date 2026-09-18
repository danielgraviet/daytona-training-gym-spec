"""Built-in coding dogfood seed (broken add) for sandbox tool-loop tests."""

from __future__ import annotations

CODING_SEED_FILES: dict[str, str] = {
    "broken.py": "def add(a, b):\n    return a - b\n",
    "test_broken.py": (
        "from broken import add\n"
        "\n"
        "result = add(1, 2)\n"
        "assert result == 3, f'expected 3, got {result}'\n"
        "print('OK')\n"
    ),
}

CODING_RUN_TESTS_COMMAND = "python test_broken.py"

CODING_DOGFOOD_PROMPT = f"""\
You are fixing a tiny Python bug in a sandbox workspace.

Files already present:
- broken.py — contains def add(a, b) that is WRONG
- test_broken.py — asserts add(1, 2) == 3

You MUST reply with exactly one JSON object per turn (no markdown fences).

To run tests:
{{"type":"tool","name":"run_tests","arguments":{{"command":"{CODING_RUN_TESTS_COMMAND}"}}}}

To overwrite a file:
{{"type":"tool","name":"write_file","arguments":{{"path":"broken.py","content":"def add(a, b):\\n    return a + b\\n"}}}}

When tests pass, finish with:
{{"type":"final","content":"fixed"}}

Start by running the tests.
"""
