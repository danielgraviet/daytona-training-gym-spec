"""Run-level metadata written once into the telemetry JSONL (cost inputs, GPU count)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_META_METRIC = "run.meta"


def append_run_meta(
    path: str | Path,
    *,
    run_id: str,
    gpu_cost_per_hour: float | None,
    num_gpus: int,
    num_steps: int | None = None,
) -> None:
    """Best-effort: never raises (telemetry must not block launch)."""
    labels: dict[str, str] = {"run_id": run_id, "num_gpus": str(int(num_gpus))}
    if gpu_cost_per_hour is not None:
        labels["gpu_cost_per_hour"] = str(float(gpu_cost_per_hour))
    if num_steps is not None:
        labels["num_steps"] = str(int(num_steps))
    record = {
        "type": "metric",
        "name": RUN_META_METRIC,
        "kind": "gauge",
        "value": 1.0,
        "labels": labels,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


def read_run_meta(metrics: Any) -> dict[str, Any]:
    """Latest ``run.meta`` labels from a sequence of MetricSample."""
    out: dict[str, Any] = {}
    for sample in metrics:
        if sample.name != RUN_META_METRIC:
            continue
        labels = sample.labels or {}
        if "gpu_cost_per_hour" in labels:
            out["gpu_cost_per_hour"] = float(labels["gpu_cost_per_hour"])
        if "num_gpus" in labels:
            out["num_gpus"] = int(labels["num_gpus"])
        if "num_steps" in labels:
            out["num_steps"] = int(labels["num_steps"])
    return out
