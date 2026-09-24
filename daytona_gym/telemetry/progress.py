"""Run progress file for the dashboard before telemetry JSONL exists."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def progress_path(runs_dir: Path | str, stem: str) -> Path:
    return Path(runs_dir) / f"{stem}.progress.json"


def write_progress(
    runs_dir: Path | str,
    stem: str,
    *,
    phase: str,
    message: str,
    append_activity: bool = True,
    **extra: Any,
) -> Path:
    """Atomically write ``runs/<stem>.progress.json`` for the live dash.

    Keeps a short ``activity`` feed (what just happened) so the UI can show
    motion even when the current phase label stays the same for minutes.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = progress_path(runs_dir, stem)
    prev = read_progress(runs_dir, stem) or {}

    activity: list[Any]
    if "activity" in extra and isinstance(extra["activity"], list):
        activity = list(extra.pop("activity"))
    else:
        activity = list(prev.get("activity") or [])
        extra.pop("activity", None)

    if append_activity:
        last = activity[-1] if activity else None
        if (
            not isinstance(last, dict)
            or last.get("phase") != phase
            or last.get("message") != message
        ):
            activity.append(
                {"t": time.time(), "phase": phase, "message": message}
            )
            activity = activity[-40:]

    # Preserve log_tail across heartbeats that omit it.
    if "log_tail" not in extra and isinstance(prev.get("log_tail"), list):
        extra = {**extra, "log_tail": prev["log_tail"]}

    payload = {
        "run_id": stem,
        "phase": phase,
        "message": message,
        "updated_at": time.time(),
        "activity": activity,
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def read_progress(runs_dir: Path | str, stem: str) -> dict[str, Any] | None:
    path = progress_path(runs_dir, stem)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def list_progress_stems(runs_dir: Path | str) -> list[str]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out: list[str] = []
    for path in runs_dir.glob("*.progress.json"):
        name = path.name
        if name.endswith(".progress.json"):
            out.append(name[: -len(".progress.json")])
    return sorted(out)
