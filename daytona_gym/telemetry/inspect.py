"""Plain-text inspector for Daytona telemetry JSONL files.

Short forms (after ``pip install -e .``):

  dg
  dg inspect
  dg runs/dogfood.jsonl
  python -m daytona_gym
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from daytona_gym.telemetry.store import (
    InMemoryTelemetryStore,
    counter_by_label,
    reconstruct_rollout,
    wall_time_decomposition,
)

_DEFAULT_PATH = Path("runs/dogfood.jsonl")
_SHORT_NAMES = {
    "sandbox.provision": "provision",
    "sandbox.seed": "seed",
    "sandbox.finalize": "finalize",
    "inference.generate": "generate",
    "rollout": "rollout",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dg",
        description="Inspect Daytona rollout telemetry (default: runs/dogfood.jsonl)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help=f"Telemetry JSONL (default: {_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--rollout",
        "-r",
        help="Rollout id (default: first in file)",
    )
    parser.add_argument(
        "--list-rollouts",
        action="store_true",
        help="List rollout ids and exit",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Legacy dense output",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Max chars of generation preview (default: 120)",
    )
    args = parser.parse_args(argv)

    path = args.path or _DEFAULT_PATH
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        print(f"tip: run from repo root, or pass a path: dg path/to.jsonl", file=sys.stderr)
        return 1

    store = InMemoryTelemetryStore.load_jsonl(path)
    rollout_ids = _rollout_ids(store)
    if args.list_rollouts:
        for rollout_id in rollout_ids:
            print(rollout_id)
        return 0

    if args.raw:
        _print_summary_raw(store, rollout_ids)
        target = args.rollout or (rollout_ids[0] if rollout_ids else None)
        if target is None:
            print("\n(no rollouts found)")
            return 0
        _print_timeline_raw(store, target)
        return 0

    target = args.rollout or (rollout_ids[0] if rollout_ids else None)
    if target is None:
        print(f"{path}  (no rollouts)")
        return 0
    _print_readable(store, path, target, preview_chars=args.preview_chars)
    return 0


def _rollout_ids(store: InMemoryTelemetryStore) -> list[str]:
    seen: list[str] = []
    for span in store.spans:
        rollout_id = str(span.attributes.get("rollout_id") or "")
        if rollout_id and rollout_id not in seen:
            seen.append(rollout_id)
    return seen


def _print_readable(
    store: InMemoryTelemetryStore,
    path: Path,
    rollout_id: str,
    *,
    preview_chars: int,
) -> None:
    timeline = reconstruct_rollout(store, rollout_id)
    outcome = _outcome_fields(store, rollout_id, timeline)
    status = outcome.get("status") or "?"
    total = timeline[0].duration_seconds if timeline else 0.0
    tool_names = Counter(
        _short_tool(str(span.attributes.get("tool") or span.name))
        for span in store.spans_for_rollout(rollout_id)
        if span.name.startswith("tool.")
    )

    print(f"{path.name}  ·  {rollout_id}  ·  {status}  ·  {_fmt_secs(total)}")
    bits: list[str] = []
    if outcome.get("reward") is not None:
        bits.append(f"reward={outcome['reward']}")
    if outcome.get("sandbox_id"):
        bits.append(f"sandbox={outcome['sandbox_id']}")
    if outcome.get("tokens") is not None:
        bits.append(f"tokens={outcome['tokens']}")
    elif outcome.get("response_tokens") is not None:
        bits.append(f"response_tokens={outcome['response_tokens']}")
    if bits:
        print("  " + "  ·  ".join(bits))
    if tool_names:
        tools = "  ".join(f"{name}×{count}" for name, count in tool_names.most_common())
        print(f"tools  {tools}")
    print()
    print("timeline")
    if not timeline:
        print("  (empty)")
        return

    for step in timeline:
        if step.name in {"rollout", "rollout.outcome"}:
            continue
        name = _short_span(step.name)
        err = f"  ! {step.error}" if step.error else ""
        print(f"  {_fmt_secs(step.offset_seconds):>7}  {name:<14}  {_fmt_secs(step.duration_seconds)}{err}")
        preview = step.attributes.get("generation_preview")
        if preview:
            for line in _preview_lines(str(preview), preview_chars):
                print(f"           {line}")

    decomposition = wall_time_decomposition(store.spans_for_rollout(rollout_id))
    parts = [(k, v) for k, v in decomposition.items() if v > 0]
    if parts:
        total_wall = sum(v for _, v in parts) or 1.0
        bits = "  ".join(f"{k} {_pct(v, total_wall)}" for k, v in parts)
        print()
        print(f"wall   {bits}")


def _outcome_fields(
    store: InMemoryTelemetryStore,
    rollout_id: str,
    timeline: list,
) -> dict[str, object]:
    """Pull status/reward/sandbox/tokens from outcome or rollout spans; derive reward if needed."""
    fields: dict[str, object] = {}
    spans = list(store.spans_for_rollout(rollout_id))
    for name in ("rollout.outcome", "rollout"):
        for span in reversed(spans):
            if span.name != name:
                continue
            attrs = span.attributes
            for key in ("status", "reward", "sandbox_id", "tokens", "response_tokens"):
                if key in attrs and attrs[key] not in (None, ""):
                    fields.setdefault(key, attrs[key])
    if "status" not in fields:
        for step in timeline:
            if step.name == "rollout" and step.attributes.get("status"):
                fields["status"] = step.attributes["status"]
                break
        if "status" not in fields:
            status_counts = counter_by_label(store.metrics_named("rollout.count"), "status")
            if status_counts:
                fields["status"] = next(iter(status_counts))
    if "reward" not in fields:
        derived = _derive_reward_from_tools(spans)
        if derived is not None:
            fields["reward"] = derived
    if "sandbox_id" not in fields:
        for span in spans:
            sid = span.attributes.get("sandbox_id")
            if sid:
                fields["sandbox_id"] = sid
                break
    return fields


def _derive_reward_from_tools(spans: list) -> float | None:
    last_ok: bool | None = None
    saw_tests = False
    for span in spans:
        if not span.name.startswith("tool."):
            continue
        tool = str(span.attributes.get("tool") or span.name.removeprefix("tool."))
        if tool != "run_tests":
            continue
        saw_tests = True
        ok = span.attributes.get("ok")
        exit_code = span.attributes.get("exit_code", 0)
        last_ok = bool(ok) and (exit_code in (0, None))
    if not saw_tests:
        return None
    return 1.0 if last_ok else 0.0


def _print_summary_raw(store: InMemoryTelemetryStore, rollout_ids: list[str]) -> None:
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


def _print_timeline_raw(store: InMemoryTelemetryStore, rollout_id: str) -> None:
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
        preview = step.attributes.get("generation_preview")
        if preview:
            print(f"           preview={preview!r}")
    decomposition = wall_time_decomposition(store.spans_for_rollout(rollout_id))
    parts = " ".join(f"{key}={value:.4f}s" for key, value in decomposition.items() if value > 0)
    if parts:
        print(f"  wall_time {parts}")


def _short_span(name: str) -> str:
    if name in _SHORT_NAMES:
        return _SHORT_NAMES[name]
    if name.startswith("tool."):
        return name.removeprefix("tool.")
    return name


def _short_tool(name: str) -> str:
    return name.removeprefix("tool.")


def _fmt_secs(value: float) -> str:
    if value < 10:
        return f"{value:.2f}s"
    return f"{value:.1f}s"


def _pct(part: float, total: float) -> str:
    return f"{100.0 * part / total:.0f}%"


def _preview_lines(preview: str, limit: int) -> list[str]:
    text = preview.replace("\\n", "↵").replace("\n", "↵")
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Prefer showing tool type snippets.
    match = re.search(r'\{"type":"[^"]+".*?\}', text)
    if match:
        text = match.group(0)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return [f"→ {text}"]


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
