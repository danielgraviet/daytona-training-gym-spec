"""Slime `--custom-generate-function-path` entry for coding dogfood.

Import path:
  daytona_gym.adapters.slime.generate_dogfood.generate
"""

from __future__ import annotations

import os
from typing import Any

from daytona_gym.adapters.slime._coding_seed import (
    CODING_RUN_TESTS_COMMAND,
    resolve_seed_profile,
)
from daytona_gym.adapters.slime.generate import generate as _generate


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _maybe_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return float(raw)


def _maybe_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return int(raw)


async def generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    path = (
        getattr(args, "daytona_telemetry_path", None)
        or os.environ.get("DAYTONA_TELEMETRY_PATH")
        or "runs/dogfood.jsonl"
    )
    args.daytona_telemetry_path = path
    if not getattr(args, "daytona_run_id", None):
        args.daytona_run_id = os.environ.get("DAYTONA_RUN_ID") or "coding_dogfood_1"
    if not getattr(args, "daytona_project_id", None):
        args.daytona_project_id = "coding-rl"
    max_turns = _maybe_int("DAYTONA_MAX_TURNS")
    if max_turns is not None:
        args.daytona_max_turns = max_turns
    elif getattr(args, "daytona_max_turns", None) is None:
        args.daytona_max_turns = 8
    if os.environ.get("DAYTONA_MAX_CONCURRENCY"):
        args.daytona_max_concurrency = int(os.environ["DAYTONA_MAX_CONCURRENCY"])
    elif getattr(args, "daytona_max_concurrency", None) is None:
        args.daytona_max_concurrency = 1

    for attr, env_name in (
        ("daytona_timeout_seconds", "DAYTONA_TIMEOUT_SECONDS"),
        ("daytona_tool_timeout_seconds", "DAYTONA_TOOL_TIMEOUT_SECONDS"),
        ("daytona_sandbox_timeout_seconds", "DAYTONA_SANDBOX_TIMEOUT_SECONDS"),
    ):
        value = _maybe_float(env_name)
        if value is not None:
            setattr(args, attr, value)

    args.daytona_return_logprob = True
    if not getattr(args, "daytona_seed_files", None):
        profile = os.environ.get("DAYTONA_SEED_PROFILE", "basic")
        args.daytona_seed_files = resolve_seed_profile(profile)
        args.daytona_seed_profile = profile

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
        f"seed_files={list((args.daytona_seed_files or {}).keys())} "
        f"timeout={getattr(args, 'daytona_timeout_seconds', None)} "
        f"tool_timeout={getattr(args, 'daytona_tool_timeout_seconds', None)} "
        f"max_turns={args.daytona_max_turns}",
        flush=True,
    )

    sample = await _generate(args, sample, sampling_params)

    meta = (getattr(sample, "metadata", None) or {}).get("daytona") or {}
    print(
        "[daytona-dogfood] done "
        f"status={meta.get('status')} sandbox={meta.get('sandbox_id')} "
        f"reward={meta.get('reward')} error={meta.get('error_code')} "
        f"tokens={len(sample.tokens) if getattr(sample, 'tokens', None) is not None else None}",
        flush=True,
    )
    return sample
