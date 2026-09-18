"""Tiny broken-add coding task used for dogfood tool-loop validation."""

from __future__ import annotations

from typing import Any

from daytona_gym.adapters.slime._coding_seed import (
    CODING_DOGFOOD_PROMPT,
    CODING_RUN_TESTS_COMMAND,
    CODING_SEED_FILES,
)

SEED_FILES = CODING_SEED_FILES
RUN_TESTS_COMMAND = CODING_RUN_TESTS_COMMAND
DOGFOOD_PROMPT = CODING_DOGFOOD_PROMPT


def inject_coding_dogfood_args(
    args: Any,
    *,
    telemetry_path: str = "runs/dogfood.jsonl",
    run_id: str = "coding_dogfood_1",
    project_id: str = "coding-rl",
    max_turns: int = 8,
    max_concurrency: int = 1,
) -> Any:
    """Attach Daytona coding-dogfood fields onto a Slime args object."""
    args.daytona_telemetry_path = telemetry_path
    args.daytona_run_id = run_id
    args.daytona_project_id = project_id
    args.daytona_max_turns = max_turns
    args.daytona_max_concurrency = max_concurrency
    args.daytona_return_logprob = True
    args.daytona_seed_files = dict(SEED_FILES)
    return args
