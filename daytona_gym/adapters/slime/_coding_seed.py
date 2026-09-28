"""Built-in coding dogfood seeds for sandbox tool-loop tests."""

from __future__ import annotations

from typing import Any

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


def _parse_label(label: Any) -> dict[str, Any] | None:
    if isinstance(label, str):
        import json

        try:
            label = json.loads(label)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(label, dict):
        return label
    return None


def seed_files_from_label(label: Any) -> dict[str, str]:
    """Extract sandbox seed files from a Harbor/coding JSONL label."""
    parsed = _parse_label(label)
    if not parsed:
        return {}
    files = parsed.get("seed_files")
    if isinstance(files, dict) and files:
        return {str(k): str(v) for k, v in files.items()}
    return {}


def run_tests_command_from_label(label: Any) -> str | None:
    """Extract per-row ``run_tests_command`` from a Harbor/coding label."""
    parsed = _parse_label(label)
    if not parsed:
        return None
    cmd = parsed.get("run_tests_command")
    if isinstance(cmd, str) and cmd.strip():
        return cmd.strip()
    return None


def seed_files_from_sample(sample: Any) -> dict[str, str]:
    """Prefer label seeds, then ``metadata.daytona.seed_files``."""
    files = seed_files_from_label(getattr(sample, "label", None))
    if files:
        return files
    meta = getattr(sample, "metadata", None) or {}
    if isinstance(meta, dict):
        daytona = meta.get("daytona") or {}
        if isinstance(daytona, dict):
            nested = daytona.get("seed_files")
            if isinstance(nested, dict) and nested:
                return {str(k): str(v) for k, v in nested.items()}
    return {}


def run_tests_command_from_sample(sample: Any) -> str | None:
    cmd = run_tests_command_from_label(getattr(sample, "label", None))
    if cmd:
        return cmd
    meta = getattr(sample, "metadata", None) or {}
    if isinstance(meta, dict):
        daytona = meta.get("daytona") or {}
        if isinstance(daytona, dict):
            nested = daytona.get("run_tests_command")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


# Default Harbor / dataset prompt: instruction + tool protocol.
CODING_AGENT_PROMPT_TEMPLATE = """\
You are a coding agent fixing a bug in a Daytona sandbox.

Task:
{instruction}

Files are already present in the workspace. Bootstrap may have run tests —
read any [environment bootstrap run_tests] output carefully.

You MUST reply with exactly one JSON object per turn (no markdown fences, no prose).

Tools:
{{"type":"tool","name":"run_tests","arguments":{{"command":"{run_tests_command}"}}}}
{{"type":"tool","name":"read_file","arguments":{{"path":"RELATIVE_PATH"}}}}
{{"type":"tool","name":"write_file","arguments":{{"path":"RELATIVE_PATH","content":"FILE_CONTENTS"}}}}

When (and only when) run_tests exits 0 / prints OK, finish with:
{{"type":"final","content":"fixed"}}

Do NOT emit final if tests still fail. Prefer write_file to fix the bug, then run_tests again.
"""


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
