"""Build and execute a local Slime×Daytona coding launch."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from daytona_gym.gym.run import TrainingRun
from daytona_gym.gym.slime_command import (
    LaunchPlan,
    build_runtime_env,
    build_train_argv,
    display_train_argv,
    expand_model_args,
    load_model_args,
    materialize_api_key_file,
    ray_job_submit_command,
)
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.telemetry.ids import new_run_id

if TYPE_CHECKING:
    from daytona_gym.gym.config import TrainConfig


def build_plan(config: TrainConfig, *, run_id: str | None = None) -> LaunchPlan:
    config.validate(require_existing_paths=False)
    rid = run_id or config.run_name or new_run_id()
    compute = config.resolved_compute()
    recipe = config.recipe
    dataset = config.dataset
    repo = compute.resolved_repo()
    telemetry = (
        Path(config.telemetry_path).expanduser().resolve()
        if config.telemetry_path
        else (repo / "runs" / f"{rid}.jsonl")
    )
    key_file = Path(
        os.environ.get("DAYTONA_API_KEY_FILE") or "/tmp/daytona_gym_api_key"
    ).expanduser()
    train_argv = build_train_argv(compute=compute, dataset=dataset, recipe=recipe)
    runtime_env = build_runtime_env(
        megatron_root=compute.megatron_root_path(),
        repo=repo,
        key_file=key_file,
        telemetry_path=telemetry,
        recipe=recipe,
        run_id=rid,
        api_url=compute.api_url,
    )
    host_env = {
        "DAYTONA_TELEMETRY_PATH": str(telemetry),
        "DAYTONA_RUN_ID": rid,
        "DAYTONA_API_KEY_FILE": str(key_file),
        "DAYTONA_API_URL": compute.api_url,
    }
    return LaunchPlan(
        run_id=rid,
        telemetry_path=telemetry,
        key_file=key_file,
        save_dir=compute.save_dir_path(),
        slime_root=compute.slime_root_path(),
        megatron_root=compute.megatron_root_path(),
        repo=repo,
        train_argv=train_argv,
        runtime_env=runtime_env,
        host_env=host_env,
        model_script=compute.model_script,
    )


def plan_to_training_run(plan: LaunchPlan, *, dry_run: bool) -> TrainingRun:
    display_argv = display_train_argv(plan.train_argv, plan.model_script)
    command = ray_job_submit_command(
        ray_address=plan.ray_address,
        runtime_env=plan.runtime_env,
        train_argv=display_argv,
        working_dir=plan.slime_root,
    )
    return TrainingRun(
        run_id=plan.run_id,
        telemetry_path=str(plan.telemetry_path),
        command=command,
        env=dict(plan.host_env),
        runtime_env=dict(plan.runtime_env),
        dry_run=dry_run,
        model_script=plan.model_script,
        inspect_hint=f"dg stats {plan.telemetry_path}",
    )


def execute_plan(
    plan: LaunchPlan,
    *,
    skip_preflight: bool = False,
    preflight_timeout_seconds: float = 90,
) -> TrainingRun:
    if shutil.which("ray") is None:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "ray not found on PATH; run on a Slime GPU host (or use launch(dry_run=True))",
        )
    if not (plan.slime_root / "train.py").is_file():
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"Slime train.py not found under {plan.slime_root}",
        )

    materialize_api_key_file(plan.key_file)

    if not skip_preflight:
        from daytona_gym.preflight import check_daytona
        import asyncio

        asyncio.run(check_daytona(timeout_seconds=preflight_timeout_seconds))

    plan.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.save_dir.exists():
        shutil.rmtree(plan.save_dir)
    plan.save_dir.mkdir(parents=True, exist_ok=True)

    model_args = load_model_args(plan.slime_root, plan.model_script)
    train_argv = expand_model_args(plan.train_argv, model_args)
    submit = ray_job_submit_command(
        ray_address=plan.ray_address,
        runtime_env=plan.runtime_env,
        train_argv=train_argv,
        working_dir=plan.slime_root,
    )

    _reset_ray(num_gpus=1)
    env = os.environ.copy()
    env.update(plan.host_env)
    env["DAYTONA_API_KEY_FILE"] = str(plan.key_file)

    proc = subprocess.run(
        submit,
        cwd=str(plan.slime_root),
        env=env,
        check=False,
    )
    return TrainingRun(
        run_id=plan.run_id,
        telemetry_path=str(plan.telemetry_path),
        command=submit,
        env=dict(plan.host_env),
        runtime_env=dict(plan.runtime_env),
        dry_run=False,
        returncode=proc.returncode,
        model_script=plan.model_script,
        inspect_hint=f"dg stats {plan.telemetry_path}",
    )


def _reset_ray(*, num_gpus: int) -> None:
    subprocess.run(
        ["bash", "-c", "pkill -9 sglang 2>/dev/null || true"],
        check=False,
    )
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    subprocess.run(
        ["bash", "-c", "pkill -9 -f 'ray::' 2>/dev/null || true"],
        check=False,
    )
    time.sleep(2)
    start = subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            "127.0.0.1",
            "--num-gpus",
            str(num_gpus),
            "--disable-usage-stats",
        ],
        check=False,
        capture_output=True,
    )
    if start.returncode != 0:
        err = (start.stderr or b"").decode("utf-8", errors="replace")[:500]
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"ray start failed: {err}",
        )
