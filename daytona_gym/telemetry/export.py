from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Protocol

_FLUSH = object()
_STOP = object()

DEFAULT_EXPORT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_EXPORT_MAX_FILES = 4
DEFAULT_EXPORT_QUEUE_SIZE = 8192
DEFAULT_EXPORT_BATCH_SIZE = 64


class TelemetryExporter(Protocol):
    def emit(self, record: dict[str, Any]) -> None: ...

    def flush(self, timeout: float | None = None) -> None: ...

    def close(self) -> None: ...


class AsyncJsonlExporter:
    """Background JSONL writer. emit() never blocks the rollout path.

    If the queue is full, the record is dropped. Rotate the active file when it
    exceeds `max_bytes`, keeping `max_files` generations on disk.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_queue: int = DEFAULT_EXPORT_QUEUE_SIZE,
        batch_size: int = DEFAULT_EXPORT_BATCH_SIZE,
        max_bytes: int | None = DEFAULT_EXPORT_MAX_BYTES,
        max_files: int = DEFAULT_EXPORT_MAX_FILES,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_files < 1:
            raise ValueError("max_files must be >= 1")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("max_bytes must be >= 1 or None")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._max_bytes = max_bytes
        self._max_files = max_files
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._handle = self._path.open("a", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._run,
            name="daytona-telemetry-export",
            daemon=True,
        )
        self._thread.start()

    def emit(self, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1

    def flush(self, timeout: float | None = 2.0) -> None:
        done = threading.Event()
        try:
            self._queue.put((_FLUSH, done), timeout=timeout)
        except queue.Full:
            return
        done.wait(timeout=timeout)

    def close(self) -> None:
        self.flush(timeout=2.0)
        try:
            self._queue.put(_STOP, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None

    @property
    def dropped(self) -> int:
        with self._drop_lock:
            return self._dropped

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    self._write(batch)
                    return
                if isinstance(item, tuple) and item and item[0] is _FLUSH:
                    self._write(batch)
                    batch = []
                    item[1].set()
                    continue
                if isinstance(item, dict):
                    batch.append(item)
                    if len(batch) >= self._batch_size:
                        self._write(batch)
                        batch = []
        finally:
            if batch:
                self._write(batch)
            if self._handle is not None and not self._handle.closed:
                self._handle.flush()

    def _write(self, batch: list[dict[str, Any]]) -> None:
        if not batch or self._handle is None or self._handle.closed:
            return
        self._maybe_rotate(sum(len(json.dumps(record, separators=(",", ":"))) + 1 for record in batch))
        for record in batch:
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._handle.flush()

    def _maybe_rotate(self, incoming_bytes: int) -> None:
        if self._max_bytes is None or self._handle is None:
            return
        current = self._handle.tell()
        if current + incoming_bytes < self._max_bytes:
            return
        self._handle.close()
        if self._max_files == 1:
            if self._path.exists():
                self._path.unlink()
        else:
            oldest = Path(f"{self._path}.{self._max_files - 1}")
            if oldest.exists():
                oldest.unlink()
            for index in range(self._max_files - 2, 0, -1):
                source = Path(f"{self._path}.{index}")
                destination = Path(f"{self._path}.{index + 1}")
                if source.exists():
                    source.replace(destination)
            if self._path.exists():
                self._path.replace(Path(f"{self._path}.1"))
        self._handle = self._path.open("a", encoding="utf-8")
