"""Plain-text inspector for Daytona telemetry JSONL files.

Usage:
  uv run python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl
  uv run python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_1
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from daytona_gym.telemetry.store import (
    InMemoryTelemetryStore,
    counter_by_label,
    reconstruct_rollout,
    wall_time_decomposition,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Daytona rollout telemetry JSONL")
    parser.add_argument("path", type=Path, help="Path to telemetry JSONL")
    parser.add_argument(
        "--rollout",
        help="Rollout id to print as a chronological timeline (default: first seen)",
    )
    parser.add_argument(
        "--list-rollouts",
        action="store_true",
        help="List rollout ids present in the file and exit",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 1

    store = InMemoryTelemetryStore.load_jsonl(args.path)
    rollout_ids = _rollout_ids(store)
    if args.list_rollouts:
        for rollout_id in rollout_ids:
            print(rollout_id)
        return 0

    _print_summary(store, rollout_ids)
    target = args.rollout or (rollout_ids[0] if rollout_ids else None)
    if target is None:
        print("\n(no rollouts found)")
        return 0
    _print_timeline(store, target)
    return 0


def _rollout_ids(store: InMemoryTelemetryStore) -> list[str]:
    seen: list[str] = []
    for span in store.spans:
        rollout_id = str(span.attributes.get("rollout_id") or "")
        if rollout_id and rollout_id not in seen:
            seen.append(rollout_id)
    return seen


def _print_summary(store: InMemoryTelemetryStore, rollout_ids: list[str]) -> None:
    status_counts = counter_by_label(store.metrics_named("rollout.count"), "status")
    durations = [sample.value for sample in store.metrics_named("rollout.duration_seconds")]
    print(f"file spans={len(store.spans)} metrics={len(store.metrics)} rollouts={len(rollout_ids)}")
    if status_counts:
        rendered = " ".join(f"{key}={int(value)}" for key, value in sorted(status_counts.items()))
        print(f"status {rendered}")
    if durations:
        print(
            "rollout.duration_seconds "
            f"n={len(durations)} "
            f"p50={_percentile(durations, 50):.4f} "
            f"p95={_percentile(durations, 95):.4f} "
            f"max={max(durations):.4f}"
        )
    tool_names = Counter(
        str(span.attributes.get("tool") or span.name)
        for span in store.spans
        if span.name.startswith("tool.")
    )
    if tool_names:
        top = ", ".join(f"{name}={count}" for name, count in tool_names.most_common(5))
        print(f"tools {top}")


def _print_timeline(store: InMemoryTelemetryStore, rollout_id: str) -> None:
    timeline = reconstruct_rollout(store, rollout_id)
    print(f"\nrollout {rollout_id}")
    if not timeline:
        print("  (no spans)")
        return
    for step in timeline:
        err = f" error={step.error}" if step.error else ""
        status = step.attributes.get("status")
        status_bit = f" status={status}" if status else ""
        print(
            f"  {step.offset_seconds:8.3f}s  {step.name:22}  "
            f"dur={step.duration_seconds:.4f}s{status_bit}{err}"
        )
    decomposition = wall_time_decomposition(store.spans_for_rollout(rollout_id))
    parts = " ".join(f"{key}={value:.4f}s" for key, value in decomposition.items() if value > 0)
    if parts:
        print(f"  wall_time {parts}")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


if __name__ == "__main__":
    raise SystemExit(main())
