"""CLI helpers for ``dg run list|status|logs|wait|stop``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from daytona_gym.gym.run import TrainingRun
from daytona_gym.telemetry import dashboard_data as data
from daytona_gym.telemetry import progress as run_progress


def _runs_dir(path: Path | None = None) -> Path:
    return Path(path or "runs").expanduser().resolve()


def _status_path(runs_dir: Path, stem: str) -> Path:
    return runs_dir / f"{stem}.status.json"


def _read_status_file(runs_dir: Path, stem: str) -> dict[str, Any] | None:
    path = _status_path(runs_dir, stem)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def list_runs(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    return data.list_run_files(_runs_dir(runs_dir))


def run_status(stem: str, runs_dir: Path | None = None) -> dict[str, Any]:
    runs_dir = _runs_dir(runs_dir)
    prog = run_progress.read_progress(runs_dir, stem) or {}
    status_file = _read_status_file(runs_dir, stem) or {}
    jsonl = runs_dir / f"{stem}.jsonl"
    item = next((r for r in data.list_run_files(runs_dir) if r.get("stem") == stem), None)
    out: dict[str, Any] = {
        "run_id": stem,
        "run_status": (item or {}).get("run_status")
        or prog.get("status")
        or status_file.get("status")
        or "unknown",
        "phase": prog.get("phase"),
        "message": prog.get("message"),
        "returncode": prog.get("returncode", status_file.get("returncode")),
        "dashboard_url": prog.get("dashboard_url") or status_file.get("dashboard_url"),
        "telemetry_path": str(jsonl) if jsonl.is_file() else None,
        "n_rollouts": (item or {}).get("n_rollouts", 0),
        "mean_reward": (item or {}).get("mean_reward"),
        "updated_at": prog.get("updated_at"),
        "progress": prog or None,
        "status_file": status_file or None,
    }
    return out


def run_logs(stem: str, runs_dir: Path | None = None, *, n: int = 40) -> str:
    runs_dir = _runs_dir(runs_dir)
    prog = run_progress.read_progress(runs_dir, stem) or {}
    lines: list[str] = []
    for a in (prog.get("activity") or [])[-n:]:
        if isinstance(a, dict):
            lines.append(f"{a.get('phase', '?')}: {a.get('message', '')}")
        else:
            lines.append(str(a))
    tail = prog.get("log_tail") or []
    if isinstance(tail, list) and tail:
        lines.append("--- log_tail ---")
        lines.extend(str(x) for x in tail[-n:])
    log_path = runs_dir / f"{stem}.detach.log"
    if log_path.is_file():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines.append(f"--- {log_path.name} ---")
            lines.extend(text.splitlines()[-n:])
        except OSError:
            pass
    return "\n".join(lines) if lines else "(no logs yet)"


def training_run_from_id(stem: str, runs_dir: Path | None = None) -> TrainingRun:
    runs_dir = _runs_dir(runs_dir)
    status = run_status(stem, runs_dir)
    telemetry = status.get("telemetry_path") or str(runs_dir / f"{stem}.jsonl")
    return TrainingRun(
        run_id=stem,
        telemetry_path=telemetry,
        command=[],
        env={},
        runtime_env={},
        dry_run=False,
        detached=True,
        status=str(status.get("run_status") or "running"),
        dashboard_url=status.get("dashboard_url"),
        returncode=status.get("returncode"),
    )


def wait_run(stem: str, runs_dir: Path | None = None, *, timeout: float | None = None) -> int:
    run = training_run_from_id(stem, runs_dir)
    run.result(timeout=timeout)
    return 0 if run.status == "completed" and int(run.returncode or 0) == 0 else 1


def stop_run(stem: str, runs_dir: Path | None = None) -> dict[str, Any]:
    """Best-effort stop: write cancel flag + kill supervised pid when known."""
    runs_dir = _runs_dir(runs_dir)
    prog = run_progress.read_progress(runs_dir, stem) or {}
    status_file = _read_status_file(runs_dir, stem) or {}
    pid = prog.get("pid") or status_file.get("pid") or prog.get("supervised_pid")
    cancelled = False
    kill_note = None
    cancel_path = runs_dir / f"{stem}.cancel"
    try:
        cancel_path.write_text(str(time.time()), encoding="utf-8")
        cancelled = True
    except OSError as exc:
        kill_note = f"cancel file failed: {exc}"
    if pid:
        try:
            import os
            import signal

            os.kill(int(pid), signal.SIGTERM)
            kill_note = f"sent SIGTERM to pid {pid}"
        except (OSError, ValueError) as exc:
            kill_note = f"kill failed: {exc}"
    run_progress.write_progress(
        runs_dir,
        stem,
        phase="failed",
        message="stop requested via dg run stop",
        status="failed",
        returncode=130,
    )
    return {"run_id": stem, "cancelled": cancelled, "note": kill_note}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "dg run — train script or run lifecycle\n\n"
            "  dg run main.py              execute a Modal-shaped train file\n"
            "  dg run list [--runs-dir DIR]\n"
            "  dg run status <id>\n"
            "  dg run logs <id> [--tail N]\n"
            "  dg run wait <id> [--timeout SEC]\n"
            "  dg run stop <id>\n"
        )
        return 0

    cmd = args[0]
    # File-exec happy path: dg run main.py  (same as uv run main.py)
    if cmd.endswith(".py") or "/" in cmd or Path(cmd).is_file():
        return _exec_train_script(cmd, args[1:])

    args = args[1:]
    runs_dir: Path | None = None
    # peel global --runs-dir
    filtered: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--runs-dir" and i + 1 < len(args):
            runs_dir = Path(args[i + 1])
            i += 2
            continue
        filtered.append(args[i])
        i += 1
    args = filtered

    if cmd in {"list", "ls"}:
        for item in list_runs(runs_dir):
            status = item.get("run_status") or "?"
            mean = item.get("mean_reward")
            mean_s = f"{mean:.3f}" if isinstance(mean, float) else "-"
            print(
                f"{item.get('stem'):<40}  {status:<10}  "
                f"n={item.get('n_rollouts', 0):<4}  reward={mean_s}"
            )
        return 0

    if cmd == "status":
        if not args:
            print("usage: dg run status <id>", file=sys.stderr)
            return 2
        info = run_status(args[0], runs_dir)
        print(json.dumps(info, indent=2, default=str))
        return 0

    if cmd == "logs":
        if not args:
            print("usage: dg run logs <id>", file=sys.stderr)
            return 2
        stem = args[0]
        n = 40
        if "--tail" in args:
            idx = args.index("--tail")
            if idx + 1 < len(args):
                n = int(args[idx + 1])
        print(run_logs(stem, runs_dir, n=n))
        return 0

    if cmd == "wait":
        if not args:
            print("usage: dg run wait <id>", file=sys.stderr)
            return 2
        stem = args[0]
        timeout = None
        if "--timeout" in args:
            idx = args.index("--timeout")
            if idx + 1 < len(args):
                timeout = float(args[idx + 1])
        return wait_run(stem, runs_dir, timeout=timeout)

    if cmd == "stop":
        if not args:
            print("usage: dg run stop <id>", file=sys.stderr)
            return 2
        print(json.dumps(stop_run(args[0], runs_dir), indent=2))
        return 0

    print(f"unknown dg run command: {cmd}", file=sys.stderr)
    return 2


def _exec_train_script(script: str, extra: list[str]) -> int:
    """Run ``script`` with uv if available, else sys.executable."""
    import os
    import shutil
    import subprocess

    path = Path(script).expanduser()
    if not path.is_file():
        print(f"dg run: file not found: {script}", file=sys.stderr)
        return 2
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "run", str(path), *extra]
    else:
        cmd = [sys.executable, str(path), *extra]
    print(f"$ {' '.join(cmd)}", flush=True)
    return int(subprocess.call(cmd, env=os.environ.copy()))
