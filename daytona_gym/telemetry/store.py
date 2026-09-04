from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from daytona_gym.telemetry.export import TelemetryExporter

DEFAULT_MAX_SPANS = 20_000
DEFAULT_MAX_METRICS = 20_000
DEFAULT_MAX_ROLLOUTS = 512


@dataclass
class StoredSpan:
    name: str
    started_at: str
    finished_at: str | None = None
    duration_seconds: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StoredSpan:
        return cls(
            name=str(payload["name"]),
            started_at=str(payload["started_at"]),
            finished_at=payload.get("finished_at"),
            duration_seconds=payload.get("duration_seconds"),
            attributes=dict(payload.get("attributes") or {}),
            error=payload.get("error"),
            seq=int(payload.get("seq") or 0),
        )


@dataclass
class MetricSample:
    name: str
    kind: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    recorded_at: str = ""
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricSample:
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            value=float(payload["value"]),
            labels={str(key): str(value) for key, value in (payload.get("labels") or {}).items()},
            recorded_at=str(payload.get("recorded_at") or ""),
            seq=int(payload.get("seq") or 0),
        )


@dataclass(frozen=True)
class TimelineStep:
    offset_seconds: float
    name: str
    duration_seconds: float
    error: str | None
    attributes: dict[str, Any]


class TelemetryStore(Protocol):
    def record_span(self, span: StoredSpan) -> None: ...

    def record_metric(self, sample: MetricSample) -> None: ...

    def spans_for_rollout(self, rollout_id: str) -> tuple[StoredSpan, ...]: ...

    def metrics_named(self, name: str) -> tuple[MetricSample, ...]: ...


class InMemoryTelemetryStore:
    """Bounded in-process ring buffer, with optional async JSONL export.

    Memory is a circular buffer (last N rollouts/spans), like an RL replay cap.
    Full history only exists if an exporter is attached. Pass `max_*=None` to
    disable a memory cap. `trace_sample_rate` drops successful traces; metrics
    and failed/aborted rollouts are always kept.
    """

    def __init__(
        self,
        *,
        max_spans: int | None = DEFAULT_MAX_SPANS,
        max_metrics: int | None = DEFAULT_MAX_METRICS,
        max_rollouts: int | None = DEFAULT_MAX_ROLLOUTS,
        exporter: TelemetryExporter | None = None,
        trace_sample_rate: float = 1.0,
    ) -> None:
        if trace_sample_rate < 0.0 or trace_sample_rate > 1.0:
            raise ValueError("trace_sample_rate must be in [0.0, 1.0]")
        self._max_spans = _require_limit(max_spans, "max_spans")
        self._max_metrics = _require_limit(max_metrics, "max_metrics")
        self._max_rollouts = _require_limit(max_rollouts, "max_rollouts")
        self._exporter = exporter
        self._trace_sample_rate = float(trace_sample_rate)
        self._spans: deque[StoredSpan] = deque()
        self._metrics: deque[MetricSample] = deque()
        self._rollout_ids: OrderedDict[str, None] = OrderedDict()
        self._seq = 0
        self._dropped_spans = 0
        self._dropped_metrics = 0
        self._dropped_rollouts = 0
        self._skipped_spans = 0
        self._lock = threading.Lock()

    def record_span(self, span: StoredSpan) -> None:
        stored = self._materialize_span(span)
        keep = self._keep_trace(stored)
        with self._lock:
            if keep:
                self._spans.append(stored)
                rollout_id = str(stored.attributes.get("rollout_id") or "")
                if rollout_id and rollout_id not in self._rollout_ids:
                    self._rollout_ids[rollout_id] = None
                self._evict_spans()
            else:
                self._skipped_spans += 1
        if keep:
            self._export({"type": "span", **stored.to_dict()})

    def record_metric(self, sample: MetricSample) -> None:
        with self._lock:
            self._seq += 1
            stored = MetricSample(
                name=sample.name,
                kind=sample.kind,
                value=sample.value,
                labels=dict(sample.labels),
                recorded_at=sample.recorded_at,
                seq=self._seq,
            )
            self._metrics.append(stored)
            if self._max_metrics is not None:
                while len(self._metrics) > self._max_metrics:
                    self._metrics.popleft()
                    self._dropped_metrics += 1
        self._export({"type": "metric", **stored.to_dict()})

    def _materialize_span(self, span: StoredSpan) -> StoredSpan:
        with self._lock:
            seq = span.seq if span.seq else 0
            if seq == 0:
                self._seq += 1
                seq = self._seq
            elif seq > self._seq:
                self._seq = seq
        return StoredSpan(
            name=span.name,
            started_at=span.started_at,
            finished_at=span.finished_at,
            duration_seconds=span.duration_seconds,
            attributes=dict(span.attributes),
            error=span.error,
            seq=seq,
        )

    def _keep_trace(self, stored: StoredSpan) -> bool:
        if stored.error:
            return True
        status = str(stored.attributes.get("status") or "")
        if status in {"failed", "aborted"}:
            return True
        rollout_id = str(stored.attributes.get("rollout_id") or "")
        return _stable_sample(rollout_id or stored.name, self._trace_sample_rate)

    def _export(self, record: dict[str, Any]) -> None:
        if self._exporter is None:
            return
        self._exporter.emit(record)

    def _evict_spans(self) -> None:
        while (
            self._max_rollouts is not None
            and len(self._rollout_ids) > self._max_rollouts
        ):
            self._drop_oldest_rollout()
        while self._max_spans is not None and len(self._spans) > self._max_spans:
            if len(self._rollout_ids) > 1:
                self._drop_oldest_rollout()
                continue
            self._spans.popleft()
            self._dropped_spans += 1

    def _drop_oldest_rollout(self) -> None:
        if not self._rollout_ids:
            if self._spans:
                self._spans.popleft()
                self._dropped_spans += 1
            return
        oldest, _unused = self._rollout_ids.popitem(last=False)
        kept: deque[StoredSpan] = deque()
        dropped = 0
        for item in self._spans:
            if str(item.attributes.get("rollout_id") or "") == oldest:
                dropped += 1
            else:
                kept.append(item)
        self._spans = kept
        self._dropped_spans += dropped
        self._dropped_rollouts += 1

    def spans_for_rollout(self, rollout_id: str) -> tuple[StoredSpan, ...]:
        with self._lock:
            matches = [
                span
                for span in self._spans
                if str(span.attributes.get("rollout_id") or "") == rollout_id
            ]
        return tuple(sorted(matches, key=lambda span: (_unix_seconds(span.started_at), span.seq)))

    def metrics_named(self, name: str) -> tuple[MetricSample, ...]:
        with self._lock:
            return tuple(sample for sample in self._metrics if sample.name == name)

    @property
    def spans(self) -> tuple[StoredSpan, ...]:
        with self._lock:
            return tuple(self._spans)

    @property
    def metrics(self) -> tuple[MetricSample, ...]:
        with self._lock:
            return tuple(self._metrics)

    @property
    def dropped_spans(self) -> int:
        with self._lock:
            return self._dropped_spans

    @property
    def dropped_metrics(self) -> int:
        with self._lock:
            return self._dropped_metrics

    @property
    def dropped_rollouts(self) -> int:
        with self._lock:
            return self._dropped_rollouts

    @property
    def skipped_spans(self) -> int:
        with self._lock:
            return self._skipped_spans

    def flush(self, timeout: float = 2.0) -> None:
        if self._exporter is not None:
            self._exporter.flush(timeout)

    def close(self) -> None:
        if self._exporter is not None:
            self._exporter.close()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "spans": [span.to_dict() for span in self._spans],
                "metrics": [sample.to_dict() for sample in self._metrics],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InMemoryTelemetryStore:
        store = cls(max_spans=None, max_metrics=None, max_rollouts=None)
        for span_payload in payload.get("spans") or []:
            store.record_span(StoredSpan.from_dict(span_payload))
        for metric_payload in payload.get("metrics") or []:
            store.record_metric(MetricSample.from_dict(metric_payload))
        return store

    def dump_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, destination.open("w", encoding="utf-8") as handle:
            for span in self._spans:
                handle.write(json.dumps({"type": "span", **span.to_dict()}, separators=(",", ":")) + "\n")
            for sample in self._metrics:
                handle.write(json.dumps({"type": "metric", **sample.to_dict()}, separators=(",", ":")) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> InMemoryTelemetryStore:
        store = cls(max_spans=None, max_metrics=None, max_rollouts=None)
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                kind = payload.pop("type", None)
                if kind == "span":
                    store.record_span(StoredSpan.from_dict(payload))
                elif kind == "metric":
                    store.record_metric(MetricSample.from_dict(payload))
        return store


def _require_limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1 or None")
    return int(value)


def _stable_sample(key: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < rate


def _unix_seconds(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def reconstruct_rollout(store: TelemetryStore, rollout_id: str) -> tuple[TimelineStep, ...]:
    """Return one rollout's spans in start order, offset from the first span."""
    spans = store.spans_for_rollout(rollout_id)
    if not spans:
        return ()
    origin = _unix_seconds(spans[0].started_at)
    steps: list[TimelineStep] = []
    for span in spans:
        steps.append(
            TimelineStep(
                offset_seconds=max(0.0, _unix_seconds(span.started_at) - origin),
                name=span.name,
                duration_seconds=float(span.duration_seconds or 0.0),
                error=span.error,
                attributes=dict(span.attributes),
            )
        )
    return tuple(steps)


_SPAN_CATEGORIES = {
    "inference.generate": "inference",
    "sandbox.provision": "sandbox",
    "sandbox.finalize": "sandbox",
    "reward.compute": "reward",
}


def _category_for(name: str) -> str:
    if name.startswith("tool."):
        return "environment"
    return _SPAN_CATEGORIES.get(name, "other")


def wall_time_decomposition(spans: Sequence[StoredSpan]) -> dict[str, float]:
    """Sum child-span durations by category. The parent `rollout` span is excluded."""
    totals: dict[str, float] = {
        "inference": 0.0,
        "sandbox": 0.0,
        "environment": 0.0,
        "reward": 0.0,
        "other": 0.0,
    }
    for span in spans:
        if span.name == "rollout":
            continue
        totals[_category_for(span.name)] += float(span.duration_seconds or 0.0)
    return totals


def counter_by_label(
    samples: Iterable[MetricSample],
    label: str,
) -> dict[str, float]:
    counts: dict[str, float] = {}
    for sample in samples:
        if sample.kind != "counter":
            continue
        key = sample.labels.get(label, "")
        counts[key] = counts.get(key, 0.0) + sample.value
    return counts
