"""Worker-side shipper: tail run telemetry and push it to the ingest service.

Design constraints (``AGENTS.md``): never block rollouts, never leak secrets.

* Runs in its own daemon thread on the launch host and only *reads* files the
  rollout path already writes — nothing is added to the rollout hot path.
* Pushes complete JSONL lines in chunks tagged with **source** byte offsets
  (``X-DG-Offset`` → ``X-DG-Next``). The server appends only when the offset
  matches what it has, so retries are idempotent; on mismatch it answers 409
  with its offset and the shipper resyncs. File truncation/rotation bumps a
  generation counter so the server restarts that run's offset at 0.
* Every chunk and progress snapshot is scrubbed of secret values (values of
  secret-looking env vars in this process + known token patterns) before it
  leaves the box.
* Network failures back off exponentially (≤60s); data stays on local disk, so
  nothing is lost while the ingest host is unreachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from daytona_gym.ingest.config import IngestConfig, valid_run_id
from daytona_gym.runtime.security import is_secret_key

DEFAULT_MAX_CHUNK_BYTES = 1024 * 1024
_TOKEN_PATTERNS = (
    re.compile(r"dtn_[A-Za-z0-9]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{16,}"),
)
_MIN_SECRET_LEN = 8


class Scrubber:
    """Replace secret *values* with ``***`` (key-based redaction happens upstream)."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        source = os.environ if env is None else env
        values = {
            v.strip()
            for k, v in source.items()
            if is_secret_key(k) and v and len(v.strip()) >= _MIN_SECRET_LEN
        }
        # Longest first so a secret containing another is replaced whole.
        self._values = sorted(values, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for value in self._values:
            if value in text:
                text = text.replace(value, "***")
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub("***", text)
        return text


Transport = Callable[[str, str, bytes, dict[str, str]], tuple[int, bytes]]


def _urllib_transport(method: str, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TelemetryShipper:
    def __init__(
        self,
        config: IngestConfig,
        *,
        telemetry_path: str | Path,
        run_id: str,
        interval_seconds: float = 2.0,
        heartbeat_seconds: float = 15.0,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        scrubber: Scrubber | None = None,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not valid_run_id(run_id):
            raise ValueError(f"invalid run id for ingest: {run_id!r}")
        self._config = config
        self._path = Path(telemetry_path)
        self._progress_path = self._path.with_name(f"{self._path.stem}.progress.json")
        self._run_id = run_id
        self._interval = interval_seconds
        self._heartbeat = heartbeat_seconds
        self._max_chunk = max_chunk_bytes
        self._scrub = scrubber or Scrubber()
        self._transport = transport or _urllib_transport
        self._clock = clock
        self._offset = 0
        self._generation = 0
        self._progress_digest: str | None = None
        self._last_contact: float | None = None
        self._backoff = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0
        self.bytes_shipped = 0
        self.skipped_lines = 0

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> TelemetryShipper:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="daytona-ingest-shipper", daemon=True
            )
            self._thread.start()
        return self

    def stop(self, *, drain_timeout: float = 15.0) -> None:
        """Stop the loop, then push whatever is left (final status included)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
            self._thread = None
        deadline = time.monotonic() + drain_timeout
        while time.monotonic() < deadline:
            try:
                more = self.ship_once()
            except Exception:  # noqa: BLE001
                return
            if not more:
                return

    # ---- one iteration -------------------------------------------------
    def ship_once(self) -> bool:
        """Push one chunk + progress. Returns True when more telemetry is pending."""
        more = self._ship_telemetry()
        self._ship_progress()
        if self._last_contact is None or self._clock() - self._last_contact >= self._heartbeat:
            self._post("POST", "heartbeat", b"", {})
        return more

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                more = self.ship_once()
                self._backoff = 0.0
                wait = 0.0 if more else self._interval
            except Exception:  # noqa: BLE001 — shipping must never crash the launch
                self.errors += 1
                self._backoff = min(60.0, max(1.0, self._backoff * 2 or 1.0))
                wait = self._backoff
            self._stop.wait(wait)

    def _ship_telemetry(self) -> bool:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return False
        if size < self._offset:
            # Truncated / rotated: new generation, start over from the top.
            self._generation += 1
            self._offset = 0
        if size == self._offset:
            return False
        with self._path.open("rb") as fh:
            fh.seek(self._offset)
            raw = fh.read(self._max_chunk)
        cut = raw.rfind(b"\n")
        if cut >= 0:
            raw = raw[: cut + 1]
            body = self._scrub(raw.decode("utf-8", errors="replace")).encode("utf-8")
            end = self._offset + len(raw)
        elif len(raw) < self._max_chunk:
            return False  # partial line still being written
        else:
            # One line larger than a chunk: skip it (empty body, offsets still
            # advance on both sides) rather than stall the stream forever.
            end = self._next_newline_after(self._offset + len(raw))
            if end is None:
                return False
            body = b""
            self.skipped_lines += 1
        start = self._offset
        status, reply = self._post(
            "POST",
            "telemetry",
            body,
            {
                "Content-Type": "application/x-ndjson",
                "X-DG-Offset": str(start),
                "X-DG-Next": str(end),
                "X-DG-Generation": str(self._generation),
            },
        )
        if status == 200:
            self._offset = end
            self.bytes_shipped += len(body)
            return end < size
        if status == 409:
            info = _json(reply)
            if int(info.get("generation", self._generation)) > self._generation:
                self._generation = int(info["generation"])
            server = int(info.get("source_offset", 0))
            # Resync to what the server has (never beyond what exists locally).
            self._offset = min(server, size)
            return True
        raise RuntimeError(f"ingest telemetry push failed: HTTP {status}")

    def _next_newline_after(self, pos: int) -> int | None:
        with self._path.open("rb") as fh:
            fh.seek(pos)
            while True:
                block = fh.read(64 * 1024)
                if not block:
                    return None
                idx = block.find(b"\n")
                if idx >= 0:
                    return pos + idx + 1
                pos += len(block)

    def _ship_progress(self) -> None:
        try:
            text = self._progress_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324 — change detection
        if digest == self._progress_digest:
            return
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return  # mid-write; next tick
        status, _ = self._post(
            "PUT",
            "progress",
            self._scrub(text).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        if status != 200:
            raise RuntimeError(f"ingest progress push failed: HTTP {status}")
        self._progress_digest = digest

    def _post(self, method: str, what: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
        url = f"{self._config.url}/v1/runs/{self._run_id}/{what}"
        status, reply = self._transport(
            method, url, body, {**headers, **self._config.auth_headers(), "User-Agent": "daytona-gym-shipper/0.1"}
        )
        if status in (200, 409):
            self._last_contact = self._clock()
        elif status in (401, 403):
            raise PermissionError("ingest rejected DAYTONA_GYM_INGEST_TOKEN")
        return status, reply


def _json(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def start_shipper_from_env(telemetry_path: str | Path, run_id: str) -> TelemetryShipper | None:
    """Start a shipper when ``DAYTONA_GYM_INGEST_URL`` + token are set; else None."""
    config = IngestConfig.from_env()
    if config is None:
        return None
    try:
        shipper = TelemetryShipper(config, telemetry_path=telemetry_path, run_id=run_id)
    except ValueError as exc:
        print(f"ingest disabled: {exc}", flush=True)
        return None
    print(f"shipping telemetry → {config.run_url(run_id)}", flush=True)
    return shipper.start()
