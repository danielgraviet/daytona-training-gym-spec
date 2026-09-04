from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self


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
