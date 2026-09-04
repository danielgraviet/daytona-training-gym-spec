from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from daytona_gym.runtime.security import sanitize_attributes
from daytona_gym.telemetry.store import MetricSample, TelemetryStore


class Metrics(Protocol):
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


class NoOpMetrics:
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        return None

    def observe(self, name: str, value: float, **labels: str) -> None:
        return None


class StoringMetrics:
    """Writes counters and observations to a TelemetryStore."""

    def __init__(self, store: TelemetryStore) -> None:
        self._store = store

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        self._record(name, value, kind="counter", **labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        self._record(name, value, kind="histogram", **labels)

    def _record(self, name: str, value: float, *, kind: str, **labels: str) -> None:
        now = datetime.now(timezone.utc)
        sanitized = sanitize_attributes(labels)
        self._store.record_metric(
            MetricSample(
                name=name,
                kind=kind,
                value=float(value),
                labels={str(key): str(item) for key, item in sanitized.items()},
                recorded_at=now.isoformat(),
            )
        )
