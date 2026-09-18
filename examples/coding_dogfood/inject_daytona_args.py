"""Copy-paste helpers to attach Daytona args onto a Slime args object."""

from __future__ import annotations

from typing import Any

from daytona_gym.adapters.slime._coding_seed import CODING_DOGFOOD_PROMPT, CODING_SEED_FILES

DOGFOOD_PROMPT = CODING_DOGFOOD_PROMPT


def inject_daytona_args(
    args: Any,
    *,
    image: str | None = None,
    snapshot: str | None = None,
    telemetry_path: str = "runs/dogfood.jsonl",
    run_id: str = "dogfood_1",
    project_id: str = "coding-rl",
    max_turns: int = 8,
    max_concurrency: int = 1,
    return_logprob: bool = True,
    seed_files: dict[str, str] | None = None,
    coding_seed: bool = False,
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
    if seed_files is not None:
        args.daytona_seed_files = dict(seed_files)
    elif coding_seed:
        args.daytona_seed_files = dict(CODING_SEED_FILES)
    # Leave daytona_generator unset so SGLangRouterGenerator is used.
    return args
