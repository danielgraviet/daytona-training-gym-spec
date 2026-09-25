"""Ingest Slime's own per-step ``perf/*`` numbers from the Ray job output.

Slime logs one dict per training step from the trainer actor, e.g. (seen live
on A100, 2026-09-25)::

    (MegatronTrainRayActor pid=4342) ... {'perf/actor_train_tflops': 91.2,
     'perf/actor_train_tok_per_s': 4851.7, 'perf/step_time': 43.88,
     'perf/wait_time_ratio': 0.9626}

The launcher already streams that output, so we parse it there (no Slime
patch, nothing in the rollout path) and append ``slime.perf.*`` gauges to the
run's telemetry JSONL, where the analysis joins them to Daytona's per-step
spans. The step index comes from the line when it names one, otherwise from
order of appearance (Slime logs once per step).
"""

from __future__ import annotations

import ast
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

METRIC_PREFIX = "slime."
_STEP_PATTERNS = (
    re.compile(r"\bperf\s+(\d+)\s*:"),
    re.compile(r"\b(?:rollout|step)(?:_id)?\s*[:=#]?\s*(\d+)\b", re.IGNORECASE),
)


def parse_perf_line(line: str) -> tuple[dict[str, float], int | None] | None:
    """Return (``perf/*`` numbers, step or None) for a Slime perf log line."""
    if "perf/" not in line:
        return None
    start, end = line.find("{"), line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = ast.literal_eval(line[start : end + 1])
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    perf = {
        str(k): float(v)
        for k, v in data.items()
        if isinstance(k, str)
        and k.startswith("perf/")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }
    if not perf:
        return None
    prefix = line[:start]
    step = None
    for pattern in _STEP_PATTERNS:
        match = pattern.search(prefix)
        if match:
            step = int(match.group(1))
            break
    return perf, step


class SlimePerfRecorder:
    """Feed Ray job output lines; appends ``slime.perf.*`` metrics per step."""

    def __init__(self, telemetry_path: str | Path, *, run_id: str) -> None:
        self._path = Path(telemetry_path)
        self._run_id = run_id
        self._next_step = 0
        self._last: dict[str, float] | None = None
        self._lock = threading.Lock()
        self.steps_recorded = 0

    def feed(self, line: str) -> None:
        """Never raises: telemetry must not break the launch."""
        try:
            parsed = parse_perf_line(line)
            if parsed is None:
                return
            perf, step = parsed
            with self._lock:
                if perf == self._last:
                    return  # same dict echoed twice (e.g. two log handlers)
                self._last = perf
                source = "log"
                if step is None:
                    step, source = self._next_step, "order"
                self._next_step = step + 1
                self._write(perf, step, source)
                self.steps_recorded += 1
        except Exception:  # noqa: BLE001
            return

    def _write(self, perf: dict[str, float], step: int, source: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        labels = {"run_id": self._run_id, "step": str(step), "step_source": source}
        lines = [
            json.dumps(
                {
                    "type": "metric",
                    "name": METRIC_PREFIX + key.replace("/", "."),
                    "kind": "gauge",
                    "value": value,
                    "labels": labels,
                    "recorded_at": now,
                },
                separators=(",", ":"),
            )
            for key, value in sorted(perf.items())
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def read_slime_perf(metrics) -> dict[str, dict[str, float]]:
    """{step: {"perf/step_time": …, …}} from stored MetricSamples."""
    out: dict[str, dict[str, float]] = {}
    for sample in metrics:
        name = getattr(sample, "name", "")
        if not name.startswith(METRIC_PREFIX + "perf."):
            continue
        step = (sample.labels or {}).get("step")
        if step is None:
            continue
        key = "perf/" + name[len(METRIC_PREFIX + "perf.") :]
        out.setdefault(str(step), {})[key] = float(sample.value)
    return out
