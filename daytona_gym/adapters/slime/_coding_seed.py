"""Built-in coding dogfood seeds for sandbox tool-loop tests."""

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

# Multi-file customer-shaped task: bug lives in util.py, tests import broken.py.
MULTIFILE_SEED_FILES: dict[str, str] = {
    "util.py": "def mul(a, b):\n    return a + b\n",
    "broken.py": (
        "from util import mul\n"
        "\n"
        "def product(a, b):\n"
        "    return mul(a, b)\n"
    ),
    "test_broken.py": (
        "from broken import product\n"
        "\n"
        "result = product(2, 3)\n"
        "assert result == 6, f'expected 6, got {result}'\n"
        "print('OK')\n"
    ),
}

SEED_PROFILES: dict[str, dict[str, str]] = {
    "basic": CODING_SEED_FILES,
    "multifile": MULTIFILE_SEED_FILES,
}

CODING_RUN_TESTS_COMMAND = "python test_broken.py"


def resolve_seed_profile(name: str | None) -> dict[str, str]:
    """Return a copy of the named seed profile (default: basic)."""
    key = (name or "basic").strip().lower() or "basic"
    files = SEED_PROFILES.get(key)
    if files is None:
        known = ", ".join(sorted(SEED_PROFILES))
        raise ValueError(f"unknown DAYTONA_SEED_PROFILE={name!r}; expected one of: {known}")
    return dict(files)

CODING_DOGFOOD_PROMPT = f"""\
You are fixing a tiny Python bug in a sandbox workspace.

Files already present:
- broken.py — def add(a, b) currently returns a - b (WRONG)
- test_broken.py — asserts add(1, 2) == 3

The environment may already have run: {CODING_RUN_TESTS_COMMAND}
Read any [environment bootstrap run_tests] output carefully.

You MUST reply with exactly one JSON object per turn (no markdown fences, no prose).

To run tests:
{{"type":"tool","name":"run_tests","arguments":{{"command":"{CODING_RUN_TESTS_COMMAND}"}}}}

To overwrite a file:
{{"type":"tool","name":"write_file","arguments":{{"path":"broken.py","content":"def add(a, b):\\n    return a + b\\n"}}}}

When (and only when) tests pass, finish with:
{{"type":"final","content":"fixed"}}

Do NOT emit final if tests still fail. The correct fix is return a + b (not a - b).
Prefer write_file to fix broken.py, then run_tests again.
"""
