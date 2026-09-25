"""Best-effort nvidia-smi sampling into telemetry metrics (optional GPU dash)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daytona_gym.telemetry.store import MetricSample, TelemetryStore


def sample_nvidia_smi() -> list[dict[str, Any]]:
    """Return per-GPU util/memory rows, or ``[]`` if nvidia-smi is unavailable."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "utilization": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


def record_gpu_metrics(store: TelemetryStore, *, run_id: str | None = None) -> int:
    """Write one nvidia-smi snapshot into ``store``. Returns samples written."""
    rows = sample_nvidia_smi()
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for row in rows:
        labels = {"gpu": str(row["index"])}
        if run_id:
            labels["run_id"] = run_id
        store.record_metric(
            MetricSample(
                name="gpu.utilization",
                kind="gauge",
                value=float(row["utilization"]),
                labels=labels,
                recorded_at=now,
            )
        )
        store.record_metric(
            MetricSample(
                name="gpu.memory_used_mb",
                kind="gauge",
                value=float(row["memory_used_mb"]),
                labels=labels,
                recorded_at=now,
            )
        )
        n += 2
    return n


class GpuMetricsSampler:
    """Background sampler that appends GPU metrics while training runs."""

    def __init__(
        self,
        store: TelemetryStore,
        *,
        interval_seconds: float = 15.0,
        run_id: str | None = None,
    ) -> None:
        self._store = store
        self._interval = max(5.0, float(interval_seconds))
        self._run_id = run_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="gpu-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                record_gpu_metrics(self._store, run_id=self._run_id)
            except Exception:  # noqa: BLE001 — never block training
                pass
            self._stop.wait(self._interval)


class JsonlGpuMetricsSampler:
    """Append nvidia-smi samples directly to a telemetry JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        interval_seconds: float = 15.0,
        run_id: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._interval = max(5.0, float(interval_seconds))
        self._run_id = run_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._loop, name="gpu-metrics-jsonl", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                now = datetime.now(timezone.utc).isoformat()
                for row in sample_nvidia_smi():
                    labels = {"gpu": str(row["index"])}
                    if self._run_id:
                        labels["run_id"] = self._run_id
                    for name, value in (
                        ("gpu.utilization", float(row["utilization"])),
                        ("gpu.memory_used_mb", float(row["memory_used_mb"])),
                    ):
                        line = json.dumps(
                            {
                                "type": "metric",
                                "name": name,
                                "kind": "gauge",
                                "value": value,
                                "labels": labels,
                                "recorded_at": now,
                            },
                            separators=(",", ":"),
                        )
                        with self._path.open("a", encoding="utf-8") as fh:
                            fh.write(line + "\n")
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self._interval)

