"""Pure builders for Slime train argv + Ray runtime_env (from coding dogfood)."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


@dataclass(frozen=True)
class LaunchPlan:
    run_id: str
    telemetry_path: Path
    key_file: Path
    save_dir: Path
    slime_root: Path
    megatron_root: Path
    repo: Path
    train_argv: list[str]
    """train.py argv *without* leading ``python3``; MODEL_ARGS slot marked."""

    runtime_env: dict[str, Any]
    host_env: dict[str, str]
    model_script: str
    ray_address: str = "http://127.0.0.1:8265"


_MODEL_ARGS_MARKER = "__DAYTONA_GYM_MODEL_ARGS__"


def build_runtime_env(
    *,
    megatron_root: Path,
    repo: Path,
    key_file: Path,
    telemetry_path: Path,
    recipe: CodingRecipe,
    run_id: str,
    api_url: str,
) -> dict[str, Any]:
    """Ray ``--runtime-env-json`` payload. Never includes ``DAYTONA_API_KEY``."""
    env_vars: dict[str, str] = {
        "PYTHONPATH": f"{megatron_root}:{repo}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "DAYTONA_API_KEY_FILE": str(key_file),
        "DAYTONA_API_URL": api_url,
        "DAYTONA_TELEMETRY_PATH": str(telemetry_path),
        "DAYTONA_MAX_CONCURRENCY": str(recipe.effective_max_concurrency()),
        "DAYTONA_SEED_CODING": "1" if recipe.seed_coding else "0",
        "DAYTONA_SEED_PROFILE": recipe.seed_profile,
        "DAYTONA_BOOTSTRAP_RUN_TESTS": "1" if recipe.bootstrap_run_tests else "0",
        "DAYTONA_BOOTSTRAP_RUN_TESTS_CMD": recipe.bootstrap_run_tests_cmd,
        "DAYTONA_REQUIRE_PASSING_TESTS": "1" if recipe.require_passing_tests else "0",
        "DAYTONA_ALLOW_ABORTED": "1" if recipe.allow_aborted else "0",
        "DAYTONA_RUN_ID": run_id,
    }
    optional: list[tuple[str, float | int | None]] = [
        ("DAYTONA_TIMEOUT_SECONDS", recipe.timeout_seconds),
        ("DAYTONA_TOOL_TIMEOUT_SECONDS", recipe.tool_timeout_seconds),
        ("DAYTONA_SANDBOX_TIMEOUT_SECONDS", recipe.sandbox_timeout_seconds),
        ("DAYTONA_MAX_TURNS", recipe.max_turns),
        ("DAYTONA_MAX_TOOLS_PER_TURN", recipe.max_tools_per_turn),
    ]
    for key, value in optional:
        if value is not None:
            env_vars[key] = str(value)

    if "DAYTONA_API_KEY" in env_vars:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            "refusing to put DAYTONA_API_KEY in Ray runtime_env",
        )
    return {"env_vars": env_vars}


def build_train_argv(
    *,
    compute: LocalSlimeCompute,
    dataset: PromptJsonlDataset,
    recipe: CodingRecipe,
) -> list[str]:
    """Slime ``train.py`` args. ``MODEL_ARGS`` inserted later via marker."""
    global_batch = recipe.global_batch_size()
    return [
        "train.py",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(compute.num_gpus),
        "--colocate",
        _MODEL_ARGS_MARKER,
        "--hf-checkpoint",
        str(compute.hf_checkpoint_path()),
        "--ref-load",
        str(compute.ref_load_path()),
        "--save",
        str(compute.save_dir_path()),
        "--save-interval",
        str(recipe.save_interval),
        "--prompt-data",
        str(dataset.resolved_path()),
        "--input-key",
        dataset.input_key,
        "--label-key",
        dataset.label_key,
        "--num-rollout",
        str(recipe.num_rollout),
        "--rollout-batch-size",
        str(recipe.batch_size),
        "--n-samples-per-prompt",
        str(recipe.n_samples),
        "--num-steps-per-rollout",
        str(recipe.num_steps_per_rollout),
        "--global-batch-size",
        str(global_batch),
        "--rollout-max-response-len",
        str(recipe.max_response_len),
        "--rollout-temperature",
        str(recipe.temperature),
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--advantage-estimator",
        "grpo",
        "--use-kl-loss",
        "--kl-loss-coef",
        "0.00",
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        "0.00",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "2048",
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-mem-fraction-static",
        str(compute.sglang_mem_fraction),
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
        "--custom-generate-function-path",
        recipe.generate_path,
        "--custom-rm-path",
        recipe.rm_path,
    ]


def expand_model_args(train_argv: list[str], model_args: list[str]) -> list[str]:
    out: list[str] = []
    for item in train_argv:
        if item == _MODEL_ARGS_MARKER:
            out.extend(model_args)
        else:
            out.append(item)
    return out


def load_model_args(slime_root: Path, model_script: str) -> list[str]:
    """Source Slime ``scripts/models/*.sh`` and print ``MODEL_ARGS``."""
    script = slime_root / "scripts" / "models" / model_script
    if not script.is_file():
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"Slime model script not found: {script}",
        )
    bash = f"""
set -e
source {json.dumps(str(script))}
printf '%s\\0' "${{MODEL_ARGS[@]}}"
"""
    proc = subprocess.run(
        ["bash", "-c", bash],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:500]
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"failed to source model script {script}: {err}",
        )
    raw = proc.stdout.decode("utf-8", errors="replace")
    parts = [p for p in raw.split("\0") if p]
    return parts


def display_train_argv(train_argv: list[str], model_script: str) -> list[str]:
    """Replace marker with a readable placeholder for dry-run display."""
    return expand_model_args(
        train_argv,
        [f"<MODEL_ARGS from scripts/models/{model_script}>"],
    )


def materialize_api_key_file(key_file: Path, api_key: str | None = None) -> Path:
    key = api_key or os.environ.get("DAYTONA_API_KEY")
    if not key:
        existing = os.environ.get("DAYTONA_API_KEY_FILE")
        if existing and Path(existing).is_file() and Path(existing).stat().st_size > 0:
            return Path(existing)
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            "DAYTONA_API_KEY is not set (or DAYTONA_API_KEY_FILE is missing/empty)",
        )
    key_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key.encode("utf-8"))
    finally:
        os.close(fd)
    return key_file


def ray_job_submit_command(
    *,
    ray_address: str,
    runtime_env: dict[str, Any],
    train_argv: list[str],
    working_dir: str | Path,
) -> list[str]:
    """Submit Slime ``train.py`` with an explicit working dir (must be Slime root)."""
    return [
        "ray",
        "job",
        "submit",
        f"--address={ray_address}",
        f"--working-dir={working_dir}",
        f"--runtime-env-json={json.dumps(runtime_env)}",
        "--",
        "python3",
        *train_argv,
    ]
