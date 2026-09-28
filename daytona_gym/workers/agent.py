"""Outbound BYO worker agent — dials into ``dg ingest``, claims jobs, trains.

Run inside the published worker image (no inbound SSH required)::

    docker run --gpus all --rm --ipc=host \\
      -v daytona-gym-models:/models --env-file ~/.daytona-gym.env \\
      docker.io/dtgraviet/daytona-gym-worker:latest \\
      daytona-gym worker --token \"$DAYTONA_GYM_INGEST_TOKEN\"

The agent long-polls ``POST /v1/workers/claim``. Safe stop: Ctrl+C finishes the
current job's wait (does not destroy the GPU host).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from daytona_gym.ingest.config import INGEST_TOKEN_ENV, INGEST_URL_ENV, IngestConfig


def _gpu_inventory() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"gpus": [], "hostname": socket.gethostname()}
    gpus = []
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append({"name": parts[0], "memory_mb": parts[1]})
    return {"gpus": gpus, "hostname": socket.gethostname()}


def _http_json(
    method: str,
    url: str,
    *,
    token: str,
    body: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "daytona-gym-worker/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", "replace")
            code = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        code = exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ingest unreachable: {exc}") from exc
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw": raw[:200]}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return code, payload


def claim_job(cfg: IngestConfig, *, worker_id: str) -> dict[str, Any] | None:
    code, data = _http_json(
        "POST",
        f"{cfg.url}/v1/workers/claim",
        token=cfg.token,
        body={"worker_id": worker_id, "gpu": _gpu_inventory()},
        timeout=70.0,
    )
    if code != 200:
        raise RuntimeError(f"claim failed HTTP {code}: {data}")
    if not data.get("id"):
        return None
    return data


def report_job(
    cfg: IngestConfig,
    job_id: str,
    *,
    status: str,
    result: dict[str, Any] | None = None,
) -> None:
    code, data = _http_json(
        "POST",
        f"{cfg.url}/v1/jobs/{job_id}/status",
        token=cfg.token,
        body={"status": status, "result": result or {}},
    )
    if code not in {200, 204}:
        print(f"warn: status report HTTP {code}: {data}", file=sys.stderr)


def _run_local_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a TrainConfig-shaped payload on this machine (LocalWorker)."""
    from daytona_gym.gym.remote_job import load_config
    from daytona_gym.gym.launch import build_plan, execute_plan
    from daytona_gym.gym.run import TrainingRun

    config = load_config(payload)
    plan = build_plan(config)
    run = TrainingRun(
        run_id=plan.run_id,
        telemetry_path=str(plan.telemetry_path),
        command=[],
        env=dict(plan.host_env),
        runtime_env=dict(plan.runtime_env),
        dry_run=False,
        returncode=None,
        model_script=plan.model_script,
        detached=False,
    )
    if payload.get("open", True):
        run.open(open_browser=False, share=False)
    finished = execute_plan(
        plan,
        skip_preflight=bool(payload.get("skip_preflight", False)),
        preflight_timeout_seconds=float(payload.get("preflight_timeout_seconds", 90)),
    )
    rc = int(finished.returncode or 0)
    return {
        "run_id": run.training_run_id,
        "returncode": rc,
        "dashboard_url": run.dashboard_url,
        "telemetry_path": run.telemetry_path,
    }


def run_loop(*, poll_seconds: float = 5.0, once: bool = False) -> int:
    cfg = IngestConfig.from_env()
    if cfg is None:
        print(
            f"set {INGEST_URL_ENV} and {INGEST_TOKEN_ENV}",
            file=sys.stderr,
        )
        return 2
    worker_id = (
        os.environ.get("DAYTONA_WORKER_ID")
        or f"agent:{socket.gethostname()}:{os.getpid()}"
    )
    print(f"daytona-gym worker {worker_id} → {cfg.url}", flush=True)
    print(f"gpu inventory: {_gpu_inventory()}", flush=True)
    while True:
        try:
            job = claim_job(cfg, worker_id=worker_id)
        except Exception as exc:  # noqa: BLE001
            print(f"claim error: {exc}", file=sys.stderr)
            time.sleep(poll_seconds)
            if once:
                return 1
            continue
        if job is None:
            if once:
                print("no jobs", flush=True)
                return 0
            time.sleep(poll_seconds)
            continue
        job_id = str(job["id"])
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        print(f"claimed {job_id}", flush=True)
        try:
            result = _run_local_payload(payload)
            status = "completed" if int(result.get("returncode") or 0) == 0 else "failed"
            report_job(cfg, job_id, status=status, result=result)
            print(f"{status} {job_id} run={result.get('run_id')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report_job(
                cfg,
                job_id,
                status="failed",
                result={"error": str(exc)[:500]},
            )
            print(f"failed {job_id}: {exc}", file=sys.stderr)
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    from daytona_gym.envfile import load_default_dotenvs

    load_default_dotenvs()
    parser = argparse.ArgumentParser(
        prog="daytona-gym worker",
        description="Outbound GPU worker agent (claims jobs from dg ingest).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=f"Ingest bearer token (default: env {INGEST_TOKEN_ENV})",
    )
    parser.add_argument(
        "--url",
        default=None,
        help=f"Ingest base URL (default: env {INGEST_URL_ENV})",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Seconds between empty claims (default 5)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim at most one job (or exit if queue empty)",
    )
    args = parser.parse_args(argv)
    if args.url:
        os.environ[INGEST_URL_ENV] = args.url.rstrip("/")
    if args.token:
        os.environ[INGEST_TOKEN_ENV] = args.token
    try:
        return run_loop(poll_seconds=max(1.0, float(args.poll)), once=bool(args.once))
    except KeyboardInterrupt:
        print("\nworker stopped", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
