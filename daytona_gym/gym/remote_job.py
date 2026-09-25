"""Run on a GPU worker: ``python -m daytona_gym.gym.remote_job /tmp/job.json``.

Prints machine-parseable markers for the laptop-side ``SshWorker``.

Detached mode (``payload["detach"]=true``): spawn a long-lived child that holds
the dashboard + training, write a status file once the dash URL is ready, print
markers to the parent, and exit so the laptop can return immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.config import TrainConfig
from daytona_gym.gym.models import SoftSlimeModel
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

_STATUS_ENV = "DAYTONA_GYM_STATUS_FILE"
_SUPERVISED_ENV = "DAYTONA_GYM_SUPERVISED"


def load_config(
    payload: dict,
    *,
    on_phase: Any | None = None,
) -> TrainConfig:
    from daytona_gym.gym.dataset import (
        INLINE_KIND,
        deserialize_dataset,
        materialize_dataset,
        materialize_inline,
    )

    raw = payload["dataset"]
    if isinstance(raw, dict) and raw.get("kind") == INLINE_KIND:
        # Dataset travelled with the job (local file on the laptop).
        runs_early = Path(payload.get("repo") or ".").expanduser() / "runs"
        dataset_cfg = materialize_inline(raw, runs_dir=runs_early)
    else:
        dataset_cfg = deserialize_dataset(raw) if isinstance(raw, dict) else raw
    recipe = CodingRecipe(**payload["recipe"])
    model = None
    compute = None
    if payload.get("model"):
        model = SoftSlimeModel(**payload["model"])
    if payload.get("compute"):
        compute = LocalSlimeCompute(**payload["compute"], repo=payload.get("repo"))
    config = TrainConfig(
        dataset=dataset_cfg,
        recipe=recipe,
        model=model,
        compute=compute,
        run_name=payload.get("run_name"),
        telemetry_path=payload.get("telemetry_path"),
        repo=payload.get("repo"),
        gpu_cost_per_hour=payload.get("gpu_cost_per_hour"),
    )
    runs = Path(payload.get("repo") or ".").expanduser() / "runs"
    kind = (
        raw.get("kind")
        if isinstance(raw, dict)
        else type(dataset_cfg).__name__
    )
    if on_phase is not None:
        on_phase(
            "dataset",
            f"Materializing dataset ({kind}) on worker…",
            detail="HuggingFace download or Harbor pull → runs/data/*.jsonl",
        )
    config.dataset = materialize_dataset(config.dataset, runs_dir=runs)
    if on_phase is not None:
        on_phase(
            "dataset",
            f"Dataset ready → {config.dataset.resolved_path()}",  # type: ignore[union-attr]
        )
    return config


def _run_supervised(payload: dict) -> int:
    """Child: progress ASAP → dataset → model → train → keep dash."""
    from daytona_gym.telemetry.ids import new_run_id
    from daytona_gym.telemetry.progress import write_progress

    status_path = Path(os.environ[_STATUS_ENV])
    repo = Path(payload.get("repo") or ".").expanduser()
    runs_dir = repo / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = str(payload.get("run_name") or new_run_id())

    def set_phase(phase: str, message: str, **extra) -> None:
        write_progress(runs_dir, stem, phase=phase, message=message, **extra)
        print(f"progress: [{phase}] {message}", flush=True)

    # Publish run_id + progress BEFORE any long download so the laptop dash lists us.
    set_phase(
        "boot",
        "Worker process started — preparing dataset / model",
        detail="Next: dataset materialize → HF download/convert → Ray/Slime",
        status="running",
        started_at=time.time(),
    )
    _write_status(
        status_path,
        {
            "run_id": stem,
            "telemetry_path": str(runs_dir / f"{stem}.jsonl"),
            "dashboard_url": None,
            "pid": os.getpid(),
        },
    )
    _print_markers(
        run_id=stem,
        telemetry=str(runs_dir / f"{stem}.jsonl"),
        dashboard=None,
        returncode=None,
        detached=True,
    )

    shipper = None
    try:
        for key, val in (payload.get("forward_env") or {}).items():
            if isinstance(key, str) and isinstance(val, str) and val:
                os.environ.setdefault(key, val)

        from daytona_gym.gym.launch import build_plan, execute_plan
        from daytona_gym.gym.run import TrainingRun

        config = load_config(payload, on_phase=set_phase)
        # Prefer the early stem so progress / jsonl stay correlated.
        if not config.run_name:
            config.run_name = stem

        plan = build_plan(config, run_id=stem)
        runs_dir = plan.telemetry_path.parent
        stem = plan.run_id

        set_phase("starting", "Plan built — preparing model weights")
        from daytona_gym.ingest.shipper import start_shipper_from_env

        shipper = start_shipper_from_env(plan.telemetry_path, plan.run_id)
        from daytona_gym.gym.launch import require_daytona_api_key

        require_daytona_api_key()

        run = TrainingRun(
            run_id=plan.run_id,
            telemetry_path=str(plan.telemetry_path),
            command=[],
            env=dict(plan.host_env),
            runtime_env=dict(plan.runtime_env),
            dry_run=False,
            returncode=None,
            model_script=plan.model_script,
            detached=True,
        )

        should_open = bool(payload.get("open", False))
        dashboard: str | None = None
        if should_open:
            dashboard = run.open(
                open_browser=bool(payload.get("open_browser", False)),
                port=int(os.environ.get("DAYTONA_GYM_DASH_PORT", "3000")),
                share=False,
            )
            _write_status(
                status_path,
                {
                    "run_id": run.training_run_id,
                    "telemetry_path": run.telemetry_path,
                    "dashboard_url": dashboard,
                    "pid": os.getpid(),
                },
            )

        if config.model is not None:
            from daytona_gym.gym.model_prep import ensure_model_ready

            ensure_model_ready(config.model, on_phase=set_phase)

        config.validate(require_existing_paths=True)
        set_phase(
            "ray_start",
            "Starting Ray / Slime…",
            detail="Preflight → Ray head → Slime submit (SGLang + Megatron boot next)",
        )

        finished = execute_plan(
            plan,
            skip_preflight=bool(payload.get("skip_preflight", False)),
            preflight_timeout_seconds=float(
                payload.get("preflight_timeout_seconds", 90)
            ),
            on_phase=set_phase,
        )
        run.returncode = finished.returncode
        run.command = finished.command
        rc = int(run.returncode or 0)
        if rc != 0:
            set_phase(
                "failed",
                f"Training exited with code {rc}",
                detail="Check the worker log / Ray job output on the GPU box",
                status="failed",
                returncode=rc,
            )
        else:
            set_phase(
                "completed",
                "Training complete",
                detail=f"returncode={rc}",
                status="completed",
                returncode=rc,
            )
        _write_status(
            status_path,
            {
                "run_id": run.training_run_id,
                "telemetry_path": run.telemetry_path,
                "dashboard_url": dashboard,
                "pid": os.getpid(),
                "done": True,
                "returncode": rc,
            },
        )
        _print_markers(
            run_id=run.training_run_id,
            telemetry=run.telemetry_path,
            dashboard=dashboard,
            returncode=rc,
            detached=True,
        )
        print(flush=True)
        if rc == 0:
            print(f"Training complete: {run.training_run_id}", flush=True)
        else:
            print(f"Training failed: {run.training_run_id} (exit={rc})", flush=True)
        if shipper is not None:
            shipper.stop()  # drain final status before blocking on the dash
            shipper = None
        if dashboard:
            run.wait_dashboard()
        return rc
    except DaytonaError as exc:
        try:
            write_progress(
                runs_dir,
                stem,
                phase="failed",
                message=f"[{exc.code}] {exc.message}",
                status="failed",
                returncode=2,
            )
        except Exception:  # noqa: BLE001
            pass
        _write_status(
            status_path,
            {
                "error": f"[{exc.code}] {exc.message}",
                "run_id": stem,
                "done": True,
                "returncode": 2,
            },
        )
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        print(f"Training failed: {stem}", flush=True)
        print("__DG_RETURNCODE__=2", flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001
        try:
            write_progress(
                runs_dir,
                stem,
                phase="failed",
                message=str(exc)[:500],
                status="failed",
                returncode=2,
            )
        except Exception:  # noqa: BLE001
            pass
        _write_status(
            status_path,
            {
                "error": str(exc),
                "run_id": stem,
                "done": True,
                "returncode": 2,
            },
        )
        print(f"daytona error: {exc}", file=sys.stderr)
        print(f"Training failed: {stem}", flush=True)
        print("__DG_RETURNCODE__=2", flush=True)
        return 2
    finally:
        if shipper is not None:
            shipper.stop()  # ship the terminal status written above


def _print_markers(
    *,
    run_id: str,
    telemetry: str,
    dashboard: str | None,
    returncode: int | None,
    detached: bool = False,
    worker_log: str | None = None,
) -> None:
    print(f"__DG_RUN_ID__={run_id}", flush=True)
    print(f"__DG_TELEMETRY__={telemetry}", flush=True)
    if dashboard:
        print(f"__DG_DASHBOARD__={dashboard}", flush=True)
    if worker_log:
        print(f"__DG_WORKER_LOG__={worker_log}", flush=True)
    if detached:
        print("__DG_DETACHED__=1", flush=True)
    if returncode is not None:
        print(f"__DG_RETURNCODE__={int(returncode)}", flush=True)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def spawn_detached_run(job_json: Path, payload: dict) -> TrainingRun:
    """Start supervised child; return handle once dashboard URL is ready."""
    from daytona_gym.gym.run import TrainingRun

    stamp = f"{os.getpid()}_{int(time.time())}"
    status = Path(f"/tmp/daytona_gym_status_{stamp}.json")
    log = Path(f"/tmp/daytona_gym_detach_{stamp}.log")
    if status.exists():
        status.unlink()

    env = os.environ.copy()
    env[_SUPERVISED_ENV] = "1"
    env[_STATUS_ENV] = str(status)

    child_payload = dict(payload)
    child_payload["detach"] = False
    child_job = Path(f"/tmp/daytona_gym_job_supervised_{stamp}.json")
    child_job.write_text(json.dumps(child_payload, indent=2), encoding="utf-8")

    log_f = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "daytona_gym.gym.remote_job", str(child_job)],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    log_f.close()

    timeout = float(payload.get("detach_ready_timeout_seconds", 1200))
    deadline = time.time() + timeout
    data: dict | None = None
    last_note = 0.0
    print(
        "waiting for run id "
        "(first boot may download+convert the model; using HF_TOKEN if set) …",
        flush=True,
    )
    while time.time() < deadline:
        now = time.time()
        if now - last_note >= 30:
            print(f"  still waiting… ({int(now - (deadline - timeout))}s) log={log}", flush=True)
            last_note = now
        if status.is_file():
            try:
                data = json.loads(status.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = None
            # Ready once we have a run_id (laptop opens localhost dash).
            if isinstance(data, dict) and data.get("run_id") and not data.get("error"):
                break
            if isinstance(data, dict) and data.get("error"):
                raise DaytonaError(
                    ErrorCode.PLATFORM_ERROR,
                    f"detached worker failed: {data['error']}",
                )
        if proc.poll() is not None and not (
            isinstance(data, dict) and data.get("run_id")
        ):
            tail = ""
            try:
                tail = log.read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                "detached worker exited before run was ready.\n" + tail,
            )
        time.sleep(0.5)

    if not isinstance(data, dict) or not data.get("run_id"):
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"timed out after {timeout:.0f}s waiting for run id "
            f"(log: {log})",
        )

    run_id = str(data["run_id"])
    telemetry = str(data.get("telemetry_path") or f"runs/{run_id}.jsonl")
    dashboard = data.get("dashboard_url")
    dashboard_s = str(dashboard) if dashboard else None
    _print_markers(
        run_id=run_id,
        telemetry=telemetry,
        dashboard=dashboard_s,
        returncode=0,
        detached=True,
        worker_log=str(log),
    )
    print(f"detached_pid={proc.pid}  log={log}", flush=True)
    return TrainingRun(
        run_id=run_id,
        telemetry_path=telemetry,
        command=[sys.executable, "-m", "daytona_gym.gym.remote_job", str(child_job)],
        env={},
        runtime_env={
            "detached": True,
            "pid": proc.pid,
            "log": str(log),
            "status_path": str(status),
            "worker_log": str(log),
        },
        dry_run=False,
        returncode=None,
        inspect_hint=f"dg stats {telemetry}"
        + (f"  |  {dashboard_s}" if dashboard_s else "  |  dg dash"),
        dashboard_url=dashboard_s,
        detached=True,
        status="running",
    )


def _spawn_detached(job_json: Path, payload: dict) -> int:
    """CLI parent entry: spawn + print markers; return process exit code."""
    try:
        spawn_detached_run(job_json, payload)
        return 0
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        print("__DG_RETURNCODE__=2", flush=True)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_json", type=Path, help="Path to job payload JSON")
    args = parser.parse_args(argv)

    payload = json.loads(args.job_json.read_text(encoding="utf-8"))
    for key, val in (payload.get("forward_env") or {}).items():
        if isinstance(key, str) and isinstance(val, str) and val:
            os.environ.setdefault(key, val)

    supervised = os.environ.get(_SUPERVISED_ENV) == "1"
    want_detach = bool(payload.get("detach", False))

    if want_detach and not supervised:
        return _spawn_detached(args.job_json, payload)

    if supervised:
        return _run_supervised(payload)

    # Attached (legacy): train to completion, then open dash.
    config = load_config(payload)
    try:
        run = config._launch_local(
            dry_run=False,
            skip_preflight=bool(payload.get("skip_preflight", False)),
            preflight_timeout_seconds=float(
                payload.get("preflight_timeout_seconds", 90)
            ),
            open=bool(payload.get("open", True)),
            open_browser=bool(payload.get("open_browser", False)),
            detach=False,
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        print("__DG_RETURNCODE__=2", flush=True)
        return 2

    _print_markers(
        run_id=run.training_run_id,
        telemetry=run.telemetry_path,
        dashboard=run.dashboard_url,
        returncode=int(run.returncode or 0),
    )
    if run.dashboard_url:
        run.wait_dashboard()
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
