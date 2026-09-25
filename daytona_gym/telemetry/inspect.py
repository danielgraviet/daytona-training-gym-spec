"""Plain-text inspector for Daytona telemetry JSONL files.

Short forms (after ``pip install -e .``):

  dg
  dg ls
  dg stats
  dg dash
  dg -r rollout_0
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
_ROLLOUT_NUM = re.compile(r"^(.*?)(\d+)$")


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
        help="List rollouts (status / reward / wall) and exit",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print aggregate status / reward / wall / tool stats and exit",
    )
    parser.add_argument(
        "--gpu-cost-per-hour",
        type=float,
        default=None,
        help="$/GPU-hour for idle-GPU cost (default: value recorded at launch)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPU count for cost (default: value recorded at launch, else 1)",
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
        _print_rollout_list(store, path, rollout_ids)
        return 0
    if args.stats:
        _print_stats(store, path, rollout_ids)
        _print_analysis(
            store,
            gpu_cost_per_hour=args.gpu_cost_per_hour,
            num_gpus=args.num_gpus,
        )
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
    return sorted(seen, key=_rollout_sort_key)


def _rollout_sort_key(rollout_id: str) -> tuple[str, int, str]:
    match = _ROLLOUT_NUM.match(rollout_id)
    if match:
        return (match.group(1), int(match.group(2)), "")
    return (rollout_id, 0, "")


def _print_rollout_list(
    store: InMemoryTelemetryStore,
    path: Path,
    rollout_ids: list[str],
) -> None:
    if not rollout_ids:
        print(f"{path.name}  (no rollouts)")
        return
    print(f"{path.name}  ·  {len(rollout_ids)} rollouts")
    for rollout_id in rollout_ids:
        timeline = reconstruct_rollout(store, rollout_id)
        outcome = _outcome_fields(store, rollout_id, timeline)
        status = outcome.get("status") or "?"
        reward = outcome.get("reward")
        reward_s = f"{reward}" if reward is not None else "-"
        err = outcome.get("error_code")
        err_s = f"  [{err}]" if err else ""
        total = timeline[0].duration_seconds if timeline else 0.0
        print(f"  {rollout_id:<12}  {status:<10}  reward={reward_s:<5}  {_fmt_secs(total)}{err_s}")


def _print_stats(
    store: InMemoryTelemetryStore,
    path: Path,
    rollout_ids: list[str],
) -> None:
    if not rollout_ids:
        print(f"{path.name}  (no rollouts)")
        return

    statuses: Counter[str] = Counter()
    rewards: Counter[object] = Counter()
    error_codes: Counter[str] = Counter()
    reward_values: list[float] = []
    walls: list[float] = []
    tools: Counter[str] = Counter()

    for rollout_id in rollout_ids:
        timeline = reconstruct_rollout(store, rollout_id)
        outcome = _outcome_fields(store, rollout_id, timeline)
        statuses[str(outcome.get("status") or "?")] += 1
        reward = outcome.get("reward")
        rewards[reward] += 1
        if outcome.get("error_code"):
            error_codes[str(outcome["error_code"])] += 1
        if isinstance(reward, (int, float)):
            reward_values.append(float(reward))
        if timeline:
            walls.append(float(timeline[0].duration_seconds))
        for span in store.spans_for_rollout(rollout_id):
            if span.name.startswith("tool."):
                tools[_short_tool(str(span.attributes.get("tool") or span.name))] += 1

    print(f"{path.name}  ·  n={len(rollout_ids)}")
    print()
    print(f"status  {'  '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}")
    reward_bits = "  ".join(
        f"{('none' if k is None else k)}={v}" for k, v in sorted(rewards.items(), key=_reward_sort_key)
    )
    mean_bit = ""
    if reward_values:
        mean_bit = f"  mean={sum(reward_values) / len(reward_values):.3f}"
    print(f"reward  {reward_bits}{mean_bit}")
    if error_codes:
        print(f"errors  {'  '.join(f'{k}={v}' for k, v in sorted(error_codes.items()))}")
    if walls:
        print(
            "wall    "
            f"p50={_fmt_secs(_percentile(walls, 50))}  "
            f"p95={_fmt_secs(_percentile(walls, 95))}  "
            f"max={_fmt_secs(max(walls))}"
        )
    if tools:
        tool_bits = "  ".join(f"{name}×{count}" for name, count in tools.most_common())
        print(f"tools   {tool_bits}")


def _print_analysis(
    store: InMemoryTelemetryStore,
    *,
    gpu_cost_per_hour: float | None,
    num_gpus: int | None,
) -> None:
    from daytona_gym.telemetry.analysis import analyze_run

    out = analyze_run(store, gpu_cost_per_hour=gpu_cost_per_hour, num_gpus=num_gpus)
    totals = out["totals"]
    if not totals["n_rollouts"]:
        return
    phase = totals["rollout_phase_seconds"]
    wait = totals["env_wait_seconds"]
    print()
    print(
        f"where time went  ({totals['n_steps']} step(s), "
        f"bound={totals['bound']})"
    )
    print(
        f"  gpu idle waiting on envs  {_fmt_secs(wait)}"
        f"  ({_pct(wait, phase)} of rollout phase)"
    )
    print(f"  straggler tax             {_fmt_secs(totals['straggler_tax_seconds'])}")
    if totals["train_phase_seconds"]:
        print(f"  train phase (derived)     {_fmt_secs(totals['train_phase_seconds'])}")
    if totals["n_failed"]:
        print(
            f"  failed-rollout waste      {_fmt_secs(totals['failed_waste_seconds'])}"
            f"  ({totals['n_failed']} rollout(s))"
        )
    prov = totals["sandbox_provision"]
    if prov["n"]:
        print(
            f"  sandbox provision         p50={_fmt_secs(prov['p50'])}"
            f"  p95={_fmt_secs(prov['p95'])}  p99={_fmt_secs(prov['p99'])}"
        )
    slow = sorted(totals["tools"].items(), key=lambda kv: kv[1]["p95"], reverse=True)[:3]
    if slow:
        bits = "  ".join(f"{name} p95={_fmt_secs(v['p95'])}" for name, v in slow)
        print(f"  slowest tools             {bits}")
    cost = out["cost"]
    if cost:
        print(
            f"  gpu cost                  ${cost['run_cost']:.2f} over rollout window, "
            f"${cost['idle_gpu_cost']:.2f} idle on envs ({cost['idle_fraction']:.0%} of spend)"
        )
    else:
        print("  gpu cost                  (pass --gpu-cost-per-hour or set "
              "TrainConfig(gpu_cost_per_hour=...))")
    if len(out["steps"]) > 1:
        print()
        print("  step  rollouts  phase     env-wait  straggler  bound")
        for s in out["steps"][-10:]:
            print(
                f"  {s['step']:>4}  {s['n_rollouts']:>8}  {_fmt_secs(s['rollout_phase_seconds']):>8}"
                f"  {_fmt_secs(s['env_wait_seconds']):>8}  {_fmt_secs(s['straggler_tax_seconds']):>9}"
                f"  {s['bound']}"
            )


def _reward_sort_key(item: tuple[object, int]) -> tuple[int, float, str]:
    key, _ = item
    if key is None:
        return (2, 0.0, "")
    if isinstance(key, (int, float)):
        return (0, float(key), "")
    return (1, 0.0, str(key))


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

    if outcome.get("reward") is not None:
        print()
        print(f"reward {outcome['reward']}")


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
            for key in ("status", "reward", "sandbox_id", "tokens", "response_tokens", "error_code"):
                if key in attrs and attrs[key] not in (None, ""):
                    fields.setdefault(key, attrs[key])
    if "status" not in fields:
        for step in timeline:
            if step.name == "rollout" and step.attributes.get("status"):
                fields["status"] = step.attributes["status"]
                break
            if step.name == "rollout" and step.attributes.get("error_code"):
                fields.setdefault("error_code", step.attributes["error_code"])
        if "status" not in fields:
            status_counts = counter_by_label(store.metrics_named("rollout.count"), "status")
            if status_counts:
                fields["status"] = next(iter(status_counts))
    if "error_code" not in fields:
        for span in spans:
            code = span.attributes.get("error_code")
            if code:
                fields["error_code"] = code
                break
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
