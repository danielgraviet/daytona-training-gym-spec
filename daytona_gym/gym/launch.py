"""Build and execute a local Slime×Daytona coding launch."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

# (phase, message, **extra) — used by the live dash before rollouts exist.
OnPhase = Callable[..., None]

# Log lines that mean "still alive" during the long Slime boot.
_BOOT_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Job submission|Submitted job|Status message: SUCCEEDED", re.I), "Ray accepted the job"),
    (re.compile(r"sglang|launching server|Scheduler.*pid", re.I), "Starting SGLang engine…"),
    (re.compile(r"Uvicorn running|Server is ready|application startup complete", re.I), "SGLang server ready"),
    (re.compile(r"Loading checkpoint|load_checkpoint|hf_validate", re.I), "Loading model weights…"),
    (re.compile(r"megatron|Megatron", re.I), "Initializing Megatron…"),
    (re.compile(r"rollout|custom.generate|generate_dogfood", re.I), "Entering rollout loop…"),
)


def build_plan(config: TrainConfig, *, run_id: str | None = None) -> LaunchPlan:
    from daytona_gym.gym.dataset import PromptJsonlDataset

    rid = run_id or config.run_name or new_run_id()
    compute = config.resolved_compute()
    recipe = config.recipe
    repo = compute.resolved_repo()
    telemetry = (
        Path(config.telemetry_path).expanduser().resolve()
        if config.telemetry_path
        else (repo / "runs" / f"{rid}.jsonl")
    )
    # Prefer already-materialized JSONL. Do not download HF/Harbor during dry-run
    # on a laptop — worker ``remote_job`` / ``validate(require_existing_paths)``
    # performs materialization.
    if isinstance(config.dataset, PromptJsonlDataset):
        dataset = config.dataset
    else:
        key = config.dataset.cache_key() or "dataset"
        pending = telemetry.parent / "data" / f"{key}.jsonl"
        if pending.is_file() and pending.stat().st_size > 0:
            dataset = PromptJsonlDataset(
                path=pending,
                input_key=config.dataset.input_key,
                label_key=config.dataset.label_key,
            )
            config.dataset = dataset
        else:
            dataset = PromptJsonlDataset(
                path=pending,
                input_key=config.dataset.input_key,
                label_key=config.dataset.label_key,
            )
    config.validate(require_existing_paths=False)
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
        "DAYTONA_NUM_GPUS": str(compute.num_gpus),
        "DAYTONA_TELEMETRY_PATH": str(telemetry),
        "DAYTONA_RUN_ID": rid,
        "DAYTONA_API_KEY_FILE": str(key_file),
        "DAYTONA_API_URL": compute.api_url,
    }
    if config.gpu_cost_per_hour is not None:
        host_env["DAYTONA_GPU_COST_PER_HOUR"] = str(float(config.gpu_cost_per_hour))
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
        inspect_hint=f"dg stats {plan.telemetry_path}  |  dg dash",
    )


def execute_plan(
    plan: LaunchPlan,
    *,
    skip_preflight: bool = False,
    preflight_timeout_seconds: float = 90,
    on_phase: OnPhase | None = None,
) -> TrainingRun:
    def note(phase: str, message: str, **extra: Any) -> None:
        if on_phase is not None:
            on_phase(phase, message, **extra)

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

        note(
            "ray_start",
            "Checking Daytona API (preflight)…",
            detail="Provisions a throwaway sandbox to verify the API key",
        )
        asyncio.run(check_daytona(timeout_seconds=preflight_timeout_seconds))
        note("ray_start", "Daytona preflight OK")

    note("ray_start", "Preparing save dir + model args…")
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

    _reset_ray(num_gpus=1, on_phase=on_phase)
    env = os.environ.copy()
    env.update(plan.host_env)
    env["DAYTONA_API_KEY_FILE"] = str(plan.key_file)

    note(
        "ray_start",
        "Submitting Slime Ray job…",
        detail="ray job submit → train.py",
    )
    from daytona_gym.telemetry.run_meta import append_run_meta

    append_run_meta(
        plan.telemetry_path,
        run_id=plan.run_id,
        gpu_cost_per_hour=_float_or_none(plan.host_env.get("DAYTONA_GPU_COST_PER_HOUR")),
        num_gpus=int(plan.host_env.get("DAYTONA_NUM_GPUS") or 1),
    )
    gpu_sampler = None
    if os.environ.get("DAYTONA_GYM_GPU_METRICS", "1") not in {"0", "false", "False"}:
        from daytona_gym.telemetry.gpu_metrics import JsonlGpuMetricsSampler

        gpu_sampler = JsonlGpuMetricsSampler(
            plan.telemetry_path,
            interval_seconds=float(os.environ.get("DAYTONA_GYM_GPU_METRICS_INTERVAL", "5")),
            run_id=plan.run_id,
        )
        gpu_sampler.start()
    try:
        returncode = _run_slime_job(
            submit,
            cwd=plan.slime_root,
            env=env,
            on_phase=on_phase,
        )
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()
    return TrainingRun(
        run_id=plan.run_id,
        telemetry_path=str(plan.telemetry_path),
        command=submit,
        env=dict(plan.host_env),
        runtime_env=dict(plan.runtime_env),
        dry_run=False,
        returncode=returncode,
        model_script=plan.model_script,
        inspect_hint=f"dg stats {plan.telemetry_path}  |  dg dash",
    )


def _reset_ray(*, num_gpus: int, on_phase: OnPhase | None = None) -> None:
    def note(message: str, **extra: Any) -> None:
        if on_phase is not None:
            on_phase("ray_start", message, **extra)

    note("Stopping leftover SGLang processes…")
    subprocess.run(
        ["bash", "-c", "pkill -9 sglang 2>/dev/null || true"],
        check=False,
    )
    note("Stopping Ray cluster…")
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    subprocess.run(
        ["bash", "-c", "pkill -9 -f 'ray::' 2>/dev/null || true"],
        check=False,
    )
    time.sleep(2)
    note(
        f"Starting Ray head ({num_gpus} GPU)…",
        detail="ray start --head --num-gpus …",
    )
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
    note("Ray head is up")


def _boot_hint_from_logs(lines: deque[str]) -> str:
    hint = "Waiting for SGLang + Megatron (often 2–3 min)…"
    for line in lines:
        for pattern, text in _BOOT_HINTS:
            if pattern.search(line):
                hint = text
    return hint


def _run_slime_job(
    submit: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    on_phase: OnPhase | None = None,
) -> int:
    """Run ``ray job submit`` while streaming progress for the live dash.

    Slime boot (SGLang + Megatron) commonly takes minutes with little output;
    we heartbeat elapsed time + a short log tail so the UI does not look stuck.
    """
    def note(phase: str, message: str, **extra: Any) -> None:
        if on_phase is not None:
            on_phase(phase, message, **extra)

    t0 = time.time()
    log_lines: deque[str] = deque(maxlen=40)
    hint = "Waiting for SGLang + Megatron (often 2–3 min)…"
    stop = threading.Event()

    def publish(*, append_activity: bool = False) -> None:
        elapsed = int(time.time() - t0)
        tail = list(log_lines)[-8:]
        note(
            "training",
            f"Slime booting · {elapsed}s — {hint}",
            detail=hint,
            elapsed_s=elapsed,
            log_tail=tail,
            append_activity=append_activity,
        )

    last_hint = hint
    note(
        "training",
        "Slime booting · 0s — Waiting for SGLang + Megatron (often 2–3 min)…",
        detail="Ray job is running; first rollout telemetry appears after boot",
        elapsed_s=0,
        log_tail=[],
        append_activity=True,
    )

    proc = subprocess.Popen(
        submit,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def reader() -> None:
        nonlocal hint, last_hint
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(raw, end="", flush=True)
            if line.strip():
                log_lines.append(line[:500])
                hint = _boot_hint_from_logs(log_lines)
                if hint != last_hint:
                    last_hint = hint
                    publish(append_activity=True)

    reader_thread = threading.Thread(target=reader, name="slime-job-log", daemon=True)
    reader_thread.start()

    def heartbeat() -> None:
        while not stop.wait(2.0):
            if proc.poll() is not None:
                break
            publish(append_activity=False)

    hb = threading.Thread(target=heartbeat, name="slime-job-hb", daemon=True)
    hb.start()

    try:
        returncode = proc.wait()
    finally:
        stop.set()
        reader_thread.join(timeout=2)
        hb.join(timeout=1)
        publish(append_activity=True)

    return int(returncode)


def _float_or_none(raw: str | None) -> float | None:
    try:
        return float(raw) if raw else None
    except ValueError:
        return None
