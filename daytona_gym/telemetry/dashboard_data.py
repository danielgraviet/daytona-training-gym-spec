"""JSON-friendly views over telemetry JSONL (for the local / SPA dashboard)."""

from __future__ import annotations

from collections import Counter
import time
from pathlib import Path
from typing import Any

from daytona_gym.telemetry import progress as run_progress
from daytona_gym.telemetry.inspect import (
    _outcome_fields,
    _rollout_ids,
    _short_span,
    _short_tool,
)
from daytona_gym.telemetry.store import (
    InMemoryTelemetryStore,
    reconstruct_rollout,
    wall_time_decomposition,
)


# Runs launched before progress files existed never record a terminal state;
# don't claim "running" once their telemetry has been quiet this long.
STALE_AFTER_SECONDS = 600.0


def infer_status_without_progress(path: Path, *, has_rollouts: bool) -> str:
    """Best-effort status when ``runs/<stem>.progress.json`` is missing."""
    if not has_rollouts:
        return "pending"
    try:
        idle = time.time() - path.stat().st_mtime
    except OSError:
        return "running"
    return "stale" if idle >= STALE_AFTER_SECONDS else "running"


def _progress_status(runs_dir: Path, stem: str) -> dict[str, Any]:
    prog = run_progress.read_progress(runs_dir, stem) or {}
    phase = str(prog.get("phase") or "")
    status = str(prog.get("status") or "")
    if not status:
        if phase == "failed" or prog.get("failed"):
            status = "failed"
        elif phase in {"done", "completed"} or prog.get("done"):
            status = "completed"
        elif phase:
            status = "running"
        else:
            status = "pending"
    return {
        "run_status": status,
        "phase": phase or None,
        "progress_message": prog.get("message"),
        "progress_updated_at": prog.get("updated_at"),
    }


def list_run_files(runs_dir: Path) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        seen.add(path.stem)
        try:
            store = InMemoryTelemetryStore.load_jsonl(path)
            ids = _rollout_ids(store)
            summary = summarize_run(store, ids)
            item = {
                "name": path.name,
                "stem": path.stem,
                "path": str(path),
                "n_rollouts": len(ids),
                "mtime": path.stat().st_mtime,
                **summary,
                **_progress_status(runs_dir, path.stem),
            }
            # Prefer terminal progress status when present.
            if item.get("run_status") in {"completed", "failed"}:
                pass
            elif run_progress.read_progress(runs_dir, path.stem) is None:
                item["run_status"] = infer_status_without_progress(path, has_rollouts=bool(ids))
            elif ids:
                item["run_status"] = "running"
            out.append(item)
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't kill the list
            out.append(
                {
                    "name": path.name,
                    "stem": path.stem,
                    "path": str(path),
                    "n_rollouts": 0,
                    "error": str(exc),
                    **_progress_status(runs_dir, path.stem),
                }
            )
    # Progress-only runs (JSONL not yet written).
    for stem in run_progress.list_progress_stems(runs_dir):
        if stem in seen:
            continue
        prog = run_progress.read_progress(runs_dir, stem) or {}
        out.append(
            {
                "name": f"{stem}.jsonl",
                "stem": stem,
                "path": str(runs_dir / f"{stem}.jsonl"),
                "n_rollouts": 0,
                "mtime": float(prog.get("updated_at") or 0),
                "statuses": {},
                "mean_reward": None,
                "pending": True,
                **_progress_status(runs_dir, stem),
            }
        )
    out.sort(key=lambda r: float(r.get("mtime") or 0), reverse=True)
    return out


def summarize_run(store: InMemoryTelemetryStore, rollout_ids: list[str]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    reward_values: list[float] = []
    walls: list[float] = []
    for rollout_id in rollout_ids:
        timeline = reconstruct_rollout(store, rollout_id)
        outcome = _outcome_fields(store, rollout_id, timeline)
        statuses[str(outcome.get("status") or "?")] += 1
        reward = outcome.get("reward")
        if isinstance(reward, (int, float)):
            reward_values.append(float(reward))
        if timeline:
            walls.append(float(timeline[0].duration_seconds))
    mean_reward = (
        sum(reward_values) / len(reward_values) if reward_values else None
    )
    return {
        "statuses": dict(statuses),
        "mean_reward": mean_reward,
        "n_rewarded": len(reward_values),
        "wall_max": max(walls) if walls else None,
        "wall_p50": _percentile(walls, 50) if walls else None,
    }


def run_detail(path: Path) -> dict[str, Any]:
    store = InMemoryTelemetryStore.load_jsonl(path)
    ids = _rollout_ids(store)
    rollouts = []
    for rollout_id in ids:
        timeline = reconstruct_rollout(store, rollout_id)
        outcome = _outcome_fields(store, rollout_id, timeline)
        tools = Counter(
            _short_tool(str(span.attributes.get("tool") or span.name))
            for span in store.spans_for_rollout(rollout_id)
            if span.name.startswith("tool.")
        )
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "status": outcome.get("status"),
                "reward": outcome.get("reward"),
                "error_code": outcome.get("error_code"),
                "wall_seconds": timeline[0].duration_seconds if timeline else None,
                "tools": dict(tools),
            }
        )
    return {
        "name": path.name,
        "stem": path.stem,
        "path": str(path),
        "summary": summarize_run(store, ids),
        "rollouts": rollouts,
    }


def rollout_detail(path: Path, rollout_id: str) -> dict[str, Any]:
    store = InMemoryTelemetryStore.load_jsonl(path)
    timeline = reconstruct_rollout(store, rollout_id)
    outcome = _outcome_fields(store, rollout_id, timeline)
    steps = []
    for step in timeline:
        if step.name in {"rollout", "rollout.outcome"}:
            continue
        steps.append(
            {
                "name": step.name,
                "short": _short_span(step.name),
                "offset_seconds": step.offset_seconds,
                "duration_seconds": step.duration_seconds,
                "error": step.error,
                "generation_preview": step.attributes.get("generation_preview"),
            }
        )
    decomp = wall_time_decomposition(store.spans_for_rollout(rollout_id))
    return {
        "name": path.name,
        "rollout_id": rollout_id,
        "status": outcome.get("status"),
        "reward": outcome.get("reward"),
        "error_code": outcome.get("error_code"),
        "wall_seconds": timeline[0].duration_seconds if timeline else None,
        "steps": steps,
        "wall_decomposition": dict(decomp),
    }


def run_charts(path: Path) -> dict[str, Any]:
    """Series for SPA charts: reward over index, status histogram, wall decomp."""
    store = InMemoryTelemetryStore.load_jsonl(path)
    ids = _rollout_ids(store)
    reward_series: list[dict[str, Any]] = []
    wall_series: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    wall_buckets: Counter[str] = Counter()
    gpu_util: list[dict[str, Any]] = []
    gpu_mem: list[dict[str, Any]] = []

    for i, rollout_id in enumerate(ids):
        timeline = reconstruct_rollout(store, rollout_id)
        outcome = _outcome_fields(store, rollout_id, timeline)
        status = str(outcome.get("status") or "?")
        status_counts[status] += 1
        reward = outcome.get("reward")
        wall = timeline[0].duration_seconds if timeline else None
        reward_series.append(
            {
                "x": i,
                "rollout_id": rollout_id,
                "y": float(reward) if isinstance(reward, (int, float)) else None,
                "status": status,
            }
        )
        wall_series.append(
            {
                "x": i,
                "rollout_id": rollout_id,
                "y": float(wall) if wall is not None else None,
            }
        )
        decomp = wall_time_decomposition(store.spans_for_rollout(rollout_id))
        for key, value in decomp.items():
            wall_buckets[str(key)] += float(value)

    for sample in store.metrics_named("gpu.utilization"):
        gpu_util.append(
            {
                "t": sample.recorded_at,
                "y": float(sample.value),
                **(sample.labels or {}),
            }
        )
    for sample in store.metrics_named("gpu.memory_used_mb"):
        gpu_mem.append(
            {
                "t": sample.recorded_at,
                "y": float(sample.value),
                **(sample.labels or {}),
            }
        )

    return {
        "name": path.name,
        "stem": path.stem,
        "n_rollouts": len(ids),
        "reward": reward_series,
        "wall": wall_series,
        "status_histogram": dict(status_counts),
        "wall_decomposition": dict(wall_buckets),
        "gpu_utilization": gpu_util,
        "gpu_memory_mb": gpu_mem,
        "summary": summarize_run(store, ids),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def run_analysis(path: Path) -> dict[str, Any]:
    """Step-level "where did the time go" analytics for the SPA."""
    from daytona_gym.telemetry.analysis import analyze_run

    out = analyze_run(InMemoryTelemetryStore.load_jsonl(path))
    out["name"] = path.name
    out["stem"] = path.stem
    return out
