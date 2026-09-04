from __future__ import annotations

from typing import Any

from daytona_gym.telemetry.export import (
    DEFAULT_EXPORT_MAX_BYTES,
    DEFAULT_EXPORT_MAX_FILES,
    AsyncJsonlExporter,
)
from daytona_gym.telemetry.metrics import Metrics, NoOpMetrics, StoringMetrics
from daytona_gym.telemetry.store import (
    DEFAULT_MAX_METRICS,
    DEFAULT_MAX_ROLLOUTS,
    DEFAULT_MAX_SPANS,
    InMemoryTelemetryStore,
    TelemetryStore,
)
from daytona_gym.telemetry.traces import NoOpTracer, StoringTracer, Tracer


def bind_telemetry(args: Any) -> tuple[TelemetryStore, Tracer, Metrics]:
    """Resolve a shared ring buffer plus optional async JSONL export.

    Memory stays capped. Set `args.daytona_telemetry_path` to stream traces to
    disk like a TRL logging callback. Disable with
    `args.daytona_telemetry_enabled = False`. Explicit tracer/metrics win.
    """
    enabled = bool(getattr(args, "daytona_telemetry_enabled", True))
    exporter = getattr(args, "daytona_telemetry_exporter", None)
    path = getattr(args, "daytona_telemetry_path", None)
    if exporter is None and path and enabled:
        exporter = AsyncJsonlExporter(
            path,
            max_bytes=_limit(args, "daytona_telemetry_export_max_bytes", DEFAULT_EXPORT_MAX_BYTES),
            max_files=int(
                getattr(args, "daytona_telemetry_export_max_files", DEFAULT_EXPORT_MAX_FILES)
                or DEFAULT_EXPORT_MAX_FILES
            ),
        )
        _try_set(args, "daytona_telemetry_exporter", exporter)

    store = getattr(args, "daytona_telemetry_store", None)
    if store is None:
        rate = float(getattr(args, "daytona_trace_sample_rate", 1.0))
        store = InMemoryTelemetryStore(
            max_spans=_limit(args, "daytona_telemetry_max_spans", DEFAULT_MAX_SPANS),
            max_metrics=_limit(args, "daytona_telemetry_max_metrics", DEFAULT_MAX_METRICS),
            max_rollouts=_limit(args, "daytona_telemetry_max_rollouts", DEFAULT_MAX_ROLLOUTS),
            exporter=exporter,
            trace_sample_rate=rate,
        )
        if enabled:
            _try_set(args, "daytona_telemetry_store", store)

    tracer: Tracer | None = getattr(args, "daytona_tracer", None)
    if tracer is None:
        tracer = StoringTracer(store) if enabled else NoOpTracer()
        _try_set(args, "daytona_tracer", tracer)

    metrics: Metrics | None = getattr(args, "daytona_metrics", None)
    if metrics is None:
        metrics = StoringMetrics(store) if enabled else NoOpMetrics()
        _try_set(args, "daytona_metrics", metrics)

    return store, tracer, metrics


def _limit(args: Any, name: str, default: int) -> int | None:
    if not hasattr(args, name):
        return default
    value = getattr(args, name)
    if value is None or int(value) == 0:
        return None
    return int(value)


def _try_set(args: Any, name: str, value: object) -> None:
    try:
        setattr(args, name, value)
    except Exception:
        return
