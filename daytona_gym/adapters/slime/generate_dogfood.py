"""Slime `--custom-generate-function-path` entry for coding dogfood.

Import path:
  daytona_gym.adapters.slime.generate_dogfood.generate
"""

from __future__ import annotations

import os
from typing import Any

from daytona_gym.adapters.slime._coding_seed import (
    CODING_RUN_TESTS_COMMAND,
    CODING_SEED_FILES,
)
from daytona_gym.adapters.slime.generate import generate as _generate


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


async def generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    path = (
        getattr(args, "daytona_telemetry_path", None)
        or os.environ.get("DAYTONA_TELEMETRY_PATH")
        or "runs/dogfood.jsonl"
    )
    args.daytona_telemetry_path = path
    if not getattr(args, "daytona_run_id", None):
        args.daytona_run_id = "coding_dogfood_1"
    if not getattr(args, "daytona_project_id", None):
        args.daytona_project_id = "coding-rl"
    if getattr(args, "daytona_max_turns", None) is None:
        args.daytona_max_turns = 8
    if os.environ.get("DAYTONA_MAX_CONCURRENCY"):
        args.daytona_max_concurrency = int(os.environ["DAYTONA_MAX_CONCURRENCY"])
    elif getattr(args, "daytona_max_concurrency", None) is None:
        args.daytona_max_concurrency = 1
    args.daytona_return_logprob = True
    if not getattr(args, "daytona_seed_files", None):
        args.daytona_seed_files = dict(CODING_SEED_FILES)

    # Always (re)apply bootstrap unless explicitly disabled. Do not rely on
    # "attribute is None" — Slime namespaces may carry stale empty values.
    if _env_flag("DAYTONA_BOOTSTRAP_RUN_TESTS", True):
        args.daytona_bootstrap_run_tests = os.environ.get(
            "DAYTONA_BOOTSTRAP_RUN_TESTS_CMD",
            CODING_RUN_TESTS_COMMAND,
        )
    else:
        args.daytona_bootstrap_run_tests = None

    print(
        "[daytona-dogfood] start "
        f"bootstrap={args.daytona_bootstrap_run_tests!r} "
        f"seed_files={list((args.daytona_seed_files or {}).keys())}",
        flush=True,
    )

    sample = await _generate(args, sample, sampling_params)

    meta = (getattr(sample, "metadata", None) or {}).get("daytona") or {}
    print(
        "[daytona-dogfood] done "
        f"status={meta.get('status')} sandbox={meta.get('sandbox_id')} "
        f"reward={meta.get('reward')} "
        f"tokens={len(sample.tokens) if getattr(sample, 'tokens', None) is not None else None}",
        flush=True,
    )
    return sample
