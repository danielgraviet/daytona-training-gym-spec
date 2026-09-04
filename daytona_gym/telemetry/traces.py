from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Protocol, Self

from daytona_gym.runtime.security import sanitize_attributes
from daytona_gym.telemetry.store import StoredSpan, TelemetryStore


class Span(Protocol):
    def set_attribute(self, key: str, value: object) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...


class Tracer(Protocol):
    def span(self, name: str, **attributes: object) -> Span: ...


class NoOpSpan:
    def set_attribute(self, key: str, value: object) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None


class NoOpTracer:
    """Cheap default tracer. Must not block rollout execution."""

    def span(self, name: str, **attributes: object) -> NoOpSpan:
        return NoOpSpan()


@dataclass
class SpanRecord:
    name: str
    attributes: dict[str, Any]
    started_at: float
    finished_at: float | None = None
    error: str | None = None


class _RecordingSpan:
    def __init__(self, tracer: RecordingTracer, record: SpanRecord) -> None:
        self._tracer = tracer
        self._record = record

    def set_attribute(self, key: str, value: object) -> None:
        self._record.attributes[key] = value

    def __enter__(self) -> Self:
        self._tracer._open_span(self._record)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self._tracer._close_span(self._record, exc_type)
        return None


@dataclass
class RecordingTracer:
    """In-memory tracer for contract tests. Not a production backend."""

    _records: list[SpanRecord] = field(default_factory=list)
    _open: list[SpanRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def span(self, name: str, **attributes: object) -> _RecordingSpan:
        record = SpanRecord(
            name=name,
            attributes=dict(attributes),
            started_at=time.perf_counter(),
        )
        return _RecordingSpan(self, record)

    def _open_span(self, record: SpanRecord) -> None:
        with self._lock:
            self._open.append(record)
            self._records.append(record)

    def _close_span(self, record: SpanRecord, exc_type: type[BaseException] | None) -> None:
        record.finished_at = time.perf_counter()
        if exc_type is not None:
            record.error = exc_type.__name__
        with self._lock:
            if record in self._open:
                self._open.remove(record)

    @property
    def records(self) -> tuple[SpanRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def open_spans(self) -> tuple[SpanRecord, ...]:
        with self._lock:
            return tuple(self._open)


class _StoringSpan:
    def __init__(self, tracer: StoringTracer, name: str, attributes: dict[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._started_at: datetime | None = None
        self._started_mono: float = 0.0
        self._seq = 0

    def set_attribute(self, key: str, value: object) -> None:
        self._attributes[key] = value

    def __enter__(self) -> Self:
        self._started_at = datetime.now(timezone.utc)
        self._started_mono = time.perf_counter()
        self._seq = self._tracer._next_seq()
        self._tracer._track(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        finished_at = datetime.now(timezone.utc)
        duration = max(0.0, time.perf_counter() - self._started_mono)
        started_at = self._started_at or finished_at
        error = exc_type.__name__ if exc_type is not None else None
        self._tracer._store.record_span(
            StoredSpan(
                name=self._name,
                started_at=started_at.isoformat(),
                finished_at=finished_at.isoformat(),
                duration_seconds=duration,
                attributes=sanitize_attributes(self._attributes),
                error=error,
                seq=self._seq,
            )
        )
        self._tracer._untrack(self)
        return None


class StoringTracer:
    """Writes closed spans to a TelemetryStore. Must stay cheap on the rollout path."""

    def __init__(self, store: TelemetryStore) -> None:
        self._store = store
        self._open: list[_StoringSpan] = []
        self._lock = threading.Lock()
        self._seq = 0

    def span(self, name: str, **attributes: object) -> _StoringSpan:
        return _StoringSpan(self, name, dict(attributes))

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _track(self, span: _StoringSpan) -> None:
        with self._lock:
            self._open.append(span)

    def _untrack(self, span: _StoringSpan) -> None:
        with self._lock:
            if span in self._open:
                self._open.remove(span)

    @property
    def open_spans(self) -> tuple[_StoringSpan, ...]:
        with self._lock:
            return tuple(self._open)
