"""Step-level rollout analytics: where did wall time and GPU time go?

Pure functions over stored spans/metrics — no I/O, no framework types. Every
derived metric ships with a human-readable definition (``DEFINITIONS``) that the
dashboard and ``dg stats`` display next to the number.

Model of a training step (Slime default, synchronous):

    |<------------- rollout phase ------------->|<-- train phase -->|
    rollout A  [gen][tool....][gen]
    rollout B  [gen][tool][gen]    [tool.........][gen]   <- straggler
               ^ first start                    ^ last end

* **Environment wait** — time inside the rollout phase when *no*
  ``inference.generate`` span is active in any rollout. The GPU has no tokens to
  generate and training cannot start, so it is idle waiting on sandboxes /
  tools / reward. This is the flagship "GPU idle waiting on environments".
* **Straggler tax** — ``last_rollout_end − median_rollout_end``: how long the
  step waited on its slowest tail.
* **Train phase (derived)** — gap between this step's last rollout end and the
  next step's first rollout start (train + weight sync + overhead). Derived from
  rollout timing until trainer-side spans are ingested.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from daytona_gym.telemetry.run_meta import read_run_meta
from daytona_gym.telemetry.slime_perf import read_slime_perf
from daytona_gym.telemetry.store import (
    InMemoryTelemetryStore,
    MetricSample,
    StoredSpan,
    _category_for,
    _unix_seconds,
)

DEFAULT_STEP_GAP_SECONDS = 1.0
FAILED_STATUSES = frozenset({"failed", "aborted", "cancelled"})

# Plain-language names + "what to do" for each part of a step. Keys stay stable
# for the API; these are what people read.
BUCKETS: dict[str, dict[str, str]] = {
    "environment": {
        "label": "waiting on sandboxes",
        "explain": "The GPU sat idle while Daytona sandboxes ran tools / tests or started up.",
        "hint": "Check the slowest tools and sandbox start-up p95 below; a prebuilt "
        "snapshot cuts start-up, and tighter tool timeouts cut stragglers.",
    },
    "inference": {
        "label": "generating",
        "explain": "The model was producing tokens for at least one rollout.",
        "hint": "Generation-bound: shorten max_response_len, cap turns, or give "
        "inference more GPU.",
    },
    "trainer": {
        "label": "training",
        "explain": "The trainer was computing gradients and updating weights.",
        "hint": "Training-bound — usually what you want; scale GPUs or batch size "
        "if steps are too slow.",
    },
    "overhead": {
        "label": "switching generate ↔ train",
        "explain": "One GPU takes turns generating and training: each step it "
        "unloads one engine, loads the other, and copies the new weights across. "
        "That switching does no useful work.",
        "hint": "The switch costs about the same every step, so do more work per "
        "step (raise batch_size / n_samples, longer tasks) — or give generation "
        "and training separate GPUs (non-colocated) to avoid it.",
    },
}

DEFINITIONS: dict[str, str] = {
    "env_wait_seconds": (
        "Time inside a step's rollout phase when no inference request was in "
        "flight in any rollout — the GPU had nothing to generate and training "
        "could not start, so it sat idle waiting on sandboxes, tools, or reward."
    ),
    "straggler_tax_seconds": (
        "Last rollout end minus median rollout end within a step: extra wall "
        "time the step spent waiting on its slowest rollouts."
    ),
    "train_phase_seconds": (
        "Derived: gap between a step's last rollout end and the next step's "
        "first rollout start (train + weight sync + overhead). Estimate until "
        "trainer-side timings are ingested."
    ),
    "bound": (
        "Per step, rollout phase only: 'environment' when environment wait is "
        "at least half of the rollout phase, otherwise 'inference'."
    ),
    "slime_step": (
        "Slime-reported per step (parsed from its perf/* log line): step_time; "
        "train = step_time × (1 − wait_time_ratio); wait = the rest."
    ),
    "overhead_seconds": (
        "Switching generate ↔ train: Slime's wait minus the Daytona-observed "
        "rollout phase — time the trainer waited on something other than "
        "rollouts. In colocated mode that is mostly unloading/loading the two "
        "engines and copying weights; also scheduling. Estimate (clamped at 0)."
    ),
    "bottleneck": (
        "Run-level: the largest bucket summed over steps — environment (GPU idle "
        "waiting on envs), inference (at least one generation in flight), and "
        "either trainer + overhead from Slime's own perf numbers when available, "
        "or trainer = derived train phase (train + offload/onload + weight sync; "
        "last step unknown) when not. Share = bucket / sum of buckets."
    ),
    "failed_waste_seconds": (
        "Inference + sandbox + tool seconds spent on rollouts that ended "
        "failed / aborted / cancelled (work that produced no training signal)."
    ),
    "idle_gpu_cost": (
        "Environment wait × $/GPU-hour × GPUs. Only shown when "
        "TrainConfig(gpu_cost_per_hour=...) was set."
    ),
    "training_step_source": (
        "'explicit' when the trainer passed a step id; 'derived' when inferred "
        "from rollout concurrency (a new step starts after all rollouts drained "
        "and stayed idle ≥ 1s)."
    ),
}


@dataclass
class RolloutFacts:
    rollout_id: str
    step: str
    step_source: str
    start: float
    end: float
    status: str
    reward: float | None
    sandbox_id: str | None
    inference_intervals: list[tuple[float, float]] = field(default_factory=list)
    decomposition: dict[str, float] = field(default_factory=dict)

    @property
    def wall(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def failed(self) -> bool:
        return self.status in FAILED_STATUSES


def _span_bounds(span: StoredSpan) -> tuple[float, float]:
    start = _unix_seconds(span.started_at)
    if span.duration_seconds is not None:
        return start, start + float(span.duration_seconds)
    if span.finished_at:
        return start, _unix_seconds(span.finished_at)
    return start, start


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def union_seconds(
    intervals: Iterable[tuple[float, float]],
    *,
    clip: tuple[float, float] | None = None,
) -> float:
    """Total length covered by ``intervals`` (overlaps counted once)."""
    spans = []
    for a, b in intervals:
        if clip is not None:
            a, b = max(a, clip[0]), min(b, clip[1])
        if b > a:
            spans.append((a, b))
    spans.sort()
    total = 0.0
    cur_a: float | None = None
    cur_b = 0.0
    for a, b in spans:
        if cur_a is None or a > cur_b:
            if cur_a is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_a is not None:
        total += cur_b - cur_a
    return total


def gaps_in(
    window: tuple[float, float], busy: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Sub-intervals of ``window`` not covered by ``busy``."""
    lo, hi = window
    merged: list[list[float]] = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in busy):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out: list[tuple[float, float]] = []
    cursor = lo
    for a, b in merged:
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if hi > cursor:
        out.append((cursor, hi))
    return out


def collect_rollouts(spans: Sequence[StoredSpan]) -> list[RolloutFacts]:
    # Key by (rollout_id, training_step): trainers may reuse sample indices
    # across steps, and merging those would corrupt every per-step number.
    by_rollout: dict[tuple[str, str], list[StoredSpan]] = defaultdict(list)
    for span in spans:
        rid = span.attributes.get("rollout_id")
        if rid:
            step_attr = span.attributes.get("training_step")
            by_rollout[(str(rid), "" if step_attr in (None, "") else str(step_attr))].append(span)

    facts: list[RolloutFacts] = []
    for (rid, _step_key), group in by_rollout.items():
        root = next((s for s in group if s.name == "rollout"), None)
        outcome = next((s for s in reversed(group) if s.name == "rollout.outcome"), None)
        timed = [s for s in group if s.name != "rollout.outcome"]
        if root is not None:
            start, end = _span_bounds(root)
        elif timed:
            bounds = [_span_bounds(s) for s in timed]
            start, end = min(b[0] for b in bounds), max(b[1] for b in bounds)
        else:
            continue

        def attr(key: str) -> Any:
            for s in (outcome, root):
                if s is not None and s.attributes.get(key) not in (None, ""):
                    return s.attributes[key]
            return None

        status = attr("status")
        if status is None:
            status = "failed" if (root is not None and root.error) else "unknown"
        reward = attr("reward")
        step = attr("training_step")
        source = attr("training_step_source") or ("explicit" if step is not None else "")

        decomposition: dict[str, float] = defaultdict(float)
        inference: list[tuple[float, float]] = []
        for s in timed:
            if s.name == "rollout":
                continue
            decomposition[_category_for(s.name)] += float(s.duration_seconds or 0.0)
            if s.name == "inference.generate":
                inference.append(_span_bounds(s))

        facts.append(
            RolloutFacts(
                rollout_id=rid,
                step="" if step is None else str(step),
                step_source=str(source),
                start=start,
                end=end,
                status=str(status),
                reward=float(reward) if isinstance(reward, (int, float)) else None,
                sandbox_id=attr("sandbox_id"),
                inference_intervals=inference,
                decomposition=dict(decomposition),
            )
        )
    facts.sort(key=lambda f: (f.start, f.rollout_id))
    return facts


def assign_missing_steps(
    rollouts: list[RolloutFacts], *, gap_seconds: float = DEFAULT_STEP_GAP_SECONDS
) -> None:
    """Offline version of ``RolloutBatchTracker`` for runs without step ids."""
    if all(r.step for r in rollouts):
        return
    step = -1
    horizon: float | None = None
    for r in sorted(rollouts, key=lambda f: f.start):
        if horizon is None or r.start - horizon >= gap_seconds:
            step += 1
        horizon = r.end if horizon is None else max(horizon, r.end)
        if not r.step:
            r.step = str(step)
            r.step_source = "derived"


def _step_sort_key(step: str) -> tuple[int, float, str]:
    try:
        return (0, float(step), step)
    except ValueError:
        return (1, 0.0, step)


def _mean_in_windows(
    samples: Sequence[tuple[float, float]], windows: Sequence[tuple[float, float]]
) -> float | None:
    values = [v for t, v in samples if any(a <= t <= b for a, b in windows)]
    return sum(values) / len(values) if values else None


def analyze_run(
    store: InMemoryTelemetryStore,
    *,
    gpu_cost_per_hour: float | None = None,
    num_gpus: int | None = None,
    step_gap_seconds: float = DEFAULT_STEP_GAP_SECONDS,
) -> dict[str, Any]:
    """Run-level + per-step analytics. Explicit cost args override run metadata."""
    spans = store.spans
    metrics: Sequence[MetricSample] = store.metrics
    meta = read_run_meta(metrics)
    rate = gpu_cost_per_hour if gpu_cost_per_hour is not None else meta.get("gpu_cost_per_hour")
    gpus = num_gpus if num_gpus is not None else meta.get("num_gpus", 1)
    hourly = (float(rate) * int(gpus)) if rate is not None else None

    rollouts = collect_rollouts(spans)
    assign_missing_steps(rollouts, gap_seconds=step_gap_seconds)

    gpu_samples = sorted(
        (_unix_seconds(s.recorded_at), float(s.value))
        for s in metrics
        if s.name == "gpu.utilization" and s.recorded_at
    )

    by_step: dict[str, list[RolloutFacts]] = defaultdict(list)
    for r in rollouts:
        by_step[r.step].append(r)
    ordered_steps = sorted(by_step, key=_step_sort_key)

    steps: list[dict[str, Any]] = []
    for idx, key in enumerate(ordered_steps):
        group = by_step[key]
        start = min(r.start for r in group)
        end = max(r.end for r in group)
        phase = max(0.0, end - start)
        inference_all = [iv for r in group for iv in r.inference_intervals]
        inference_busy = union_seconds(inference_all, clip=(start, end))
        wait_windows = gaps_in((start, end), inference_all)
        env_wait = sum(b - a for a, b in wait_windows)
        ends = [r.end for r in group]
        straggler = end - percentile(ends, 50)
        critical = max(group, key=lambda r: r.end)
        next_start = (
            min(r.start for r in by_step[ordered_steps[idx + 1]])
            if idx + 1 < len(ordered_steps)
            else None
        )
        train_phase = max(0.0, next_start - end) if next_start is not None else None
        sources = {r.step_source for r in group if r.step_source}
        steps.append(
            {
                "step": key,
                "training_step_source": sources.pop() if len(sources) == 1 else "mixed",
                "start": start,
                "end": end,
                "n_rollouts": len(group),
                "n_failed": sum(1 for r in group if r.failed),
                "rollout_phase_seconds": phase,
                "inference_busy_seconds": inference_busy,
                "env_wait_seconds": env_wait,
                "env_wait_windows": wait_windows,
                "env_wait_fraction": (env_wait / phase) if phase > 0 else 0.0,
                "bound": "environment" if phase > 0 and env_wait >= 0.5 * phase else "inference",
                "straggler_tax_seconds": straggler,
                "critical_rollout_id": critical.rollout_id,
                "critical_rollout_decomposition": critical.decomposition,
                "train_phase_seconds": train_phase,
                "gpu_util_mean": _mean_in_windows(gpu_samples, [(start, end)]),
                "gpu_util_mean_during_env_wait": _mean_in_windows(gpu_samples, wait_windows),
                "idle_gpu_cost": (env_wait / 3600.0 * hourly) if hourly is not None else None,
            }
        )

    slime = read_slime_perf(metrics)
    for s in steps:
        perf = slime.get(s["step"])
        if not perf or "perf/step_time" not in perf:
            s["slime"] = None
            continue
        step_time = perf["perf/step_time"]
        ratio = min(1.0, max(0.0, perf.get("perf/wait_time_ratio", 0.0)))
        wait = step_time * ratio
        s["slime"] = {
            "step_time": step_time,
            "wait_time_ratio": ratio,
            "train_seconds": step_time - wait,
            "wait_seconds": wait,
            "overhead_seconds": max(0.0, wait - s["rollout_phase_seconds"]),
            "perf": perf,
        }

    provision = [
        float(s.duration_seconds or 0.0) for s in spans if s.name == "sandbox.provision"
    ]
    tools: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        if s.name.startswith("tool."):
            tools[s.name[len("tool.") :]].append(float(s.duration_seconds or 0.0))

    failed = [r for r in rollouts if r.failed]
    failed_waste = sum(
        r.decomposition.get(k, 0.0) for r in failed for k in ("inference", "sandbox", "environment")
    )
    run_start = min((r.start for r in rollouts), default=None)
    run_end = max((r.end for r in rollouts), default=None)
    run_wall = (run_end - run_start) if run_start is not None and run_end is not None else 0.0
    env_wait_total = sum(s["env_wait_seconds"] for s in steps)
    phase_total = sum(s["rollout_phase_seconds"] for s in steps)
    train_total = sum(s["train_phase_seconds"] or 0.0 for s in steps)
    inference_total = sum(s["inference_busy_seconds"] for s in steps)
    slime_steps = [s for s in steps if s.get("slime")]
    if slime_steps:
        # Trainer-reported: split the old "trainer" gap into real training
        # time and wait overhead (engine swaps / weight sync).
        breakdown = {
            "environment": sum(s["env_wait_seconds"] for s in slime_steps),
            "inference": sum(s["inference_busy_seconds"] for s in slime_steps),
            "trainer": sum(s["slime"]["train_seconds"] for s in slime_steps),
            "overhead": sum(s["slime"]["overhead_seconds"] for s in slime_steps),
        }
        trainer_source = "slime"
    else:
        breakdown = {
            "environment": env_wait_total,
            "inference": inference_total,
            "trainer": train_total,
        }
        trainer_source = "derived"
    breakdown_sum = sum(breakdown.values())
    bottleneck = max(breakdown, key=breakdown.__getitem__) if breakdown_sum > 0 else "unknown"

    totals = {
        "n_rollouts": len(rollouts),
        "n_steps": len(steps),
        "n_failed": len(failed),
        "run_wall_seconds": run_wall,
        "rollout_phase_seconds": phase_total,
        "train_phase_seconds": train_total,
        "env_wait_seconds": env_wait_total,
        "env_wait_fraction_of_run": (env_wait_total / run_wall) if run_wall > 0 else 0.0,
        "straggler_tax_seconds": sum(s["straggler_tax_seconds"] for s in steps),
        "failed_waste_seconds": failed_waste,
        "rollout_bound": (
            "environment"
            if phase_total > 0 and env_wait_total >= 0.5 * phase_total
            else ("inference" if phase_total > 0 else "unknown")
        ),
        "bottleneck": bottleneck,
        "trainer_source": trainer_source,
        "slime_step_time_seconds": sum(s["slime"]["step_time"] for s in slime_steps),
        "bottleneck_share": (
            breakdown[bottleneck] / breakdown_sum if breakdown_sum > 0 else 0.0
        ),
        "bottleneck_label": BUCKETS.get(bottleneck, {}).get("label", bottleneck),
        "bottleneck_explain": BUCKETS.get(bottleneck, {}).get("explain", ""),
        "bottleneck_hint": BUCKETS.get(bottleneck, {}).get("hint", ""),
        "time_breakdown": {
            k: {
                "seconds": v,
                "share": (v / breakdown_sum) if breakdown_sum > 0 else 0.0,
                "label": BUCKETS.get(k, {}).get("label", k),
            }
            for k, v in breakdown.items()
        },
        "sandbox_provision": {
            "n": len(provision),
            "p50": percentile(provision, 50),
            "p95": percentile(provision, 95),
            "p99": percentile(provision, 99),
        },
        "tools": {
            name: {
                "n": len(vals),
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
                "total_seconds": sum(vals),
            }
            for name, vals in sorted(tools.items())
        },
    }
    cost: dict[str, Any] | None = None
    if hourly is not None:
        run_cost = run_wall / 3600.0 * hourly
        idle_cost = env_wait_total / 3600.0 * hourly
        cost = {
            "gpu_cost_per_hour": float(rate),  # type: ignore[arg-type]
            "num_gpus": int(gpus),
            "run_cost": run_cost,
            "idle_gpu_cost": idle_cost,
            "idle_fraction": (idle_cost / run_cost) if run_cost > 0 else 0.0,
            "straggler_cost": totals["straggler_tax_seconds"] / 3600.0 * hourly,
            "failed_waste_cost": failed_waste / 3600.0 * hourly,
        }

    return {
        "totals": totals,
        "cost": cost,
        "steps": steps,
        "rollouts": [
            {
                "rollout_id": r.rollout_id,
                "step": r.step,
                "start": r.start,
                "end": r.end,
                "wall": r.wall,
                "status": r.status,
                "reward": r.reward,
                "decomposition": r.decomposition,
                "inference_intervals": r.inference_intervals,
            }
            for r in rollouts
        ],
        "gpu_utilization": [{"t": t, "y": v} for t, v in gpu_samples],
        "definitions": DEFINITIONS,
    }
