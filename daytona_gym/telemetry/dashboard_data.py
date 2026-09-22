"""JSON/HTML-friendly views over telemetry JSONL (for the basic local dashboard)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

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


def list_run_files(runs_dir: Path) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            store = InMemoryTelemetryStore.load_jsonl(path)
            ids = _rollout_ids(store)
            summary = summarize_run(store, ids)
            out.append(
                {
                    "name": path.name,
                    "stem": path.stem,
                    "path": str(path),
                    "n_rollouts": len(ids),
                    "mtime": path.stat().st_mtime,
                    **summary,
                }
            )
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't kill the list
            out.append(
                {
                    "name": path.name,
                    "stem": path.stem,
                    "path": str(path),
                    "n_rollouts": 0,
                    "error": str(exc),
                }
            )
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
