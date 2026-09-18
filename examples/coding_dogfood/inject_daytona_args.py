"""Copy-paste helpers to attach Daytona args onto a Slime args object."""

from __future__ import annotations

from typing import Any


def inject_daytona_args(
    args: Any,
    *,
    image: str | None = None,
    snapshot: str | None = None,
    telemetry_path: str = "runs/dogfood.jsonl",
    run_id: str = "dogfood_1",
    project_id: str = "coding-rl",
    max_turns: int = 6,
    max_concurrency: int = 2,
    return_logprob: bool = True,
) -> Any:
    """Set Daytona fields Slime's custom generate hook reads from `args`."""
    if image is not None:
        args.daytona_image = image
    if snapshot is not None:
        args.daytona_snapshot = snapshot
    args.daytona_telemetry_path = telemetry_path
    args.daytona_run_id = run_id
    args.daytona_project_id = project_id
    args.daytona_max_turns = max_turns
    args.daytona_max_concurrency = max_concurrency
    args.daytona_return_logprob = return_logprob
    # Leave daytona_generator unset so SGLangRouterGenerator is used.
    return args


# Example prompt for the tiny broken-add task.
DOGFOOD_PROMPT = """\
Fix the failing tests in the workspace.
Use tools when needed. Prefer run_tests with command "pytest".
When finished, respond with a JSON final message:
{"type":"final","content":"fixed"}
"""
