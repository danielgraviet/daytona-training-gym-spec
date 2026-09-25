"""Derive rollout batch / training-step ids when the trainer does not pass them.

Slime's custom-generate hook receives ``(args, sample, sampling_params)`` only, so
the training step is not directly visible. In Slime's default (synchronous)
mode every step drains all in-flight rollouts before training, so a rollout that
starts after the in-flight count has sat at zero for at least ``min_gap_seconds``
begins a new batch. The gap guard keeps millisecond dips mid-batch (staggered
submission, tight concurrency caps) from splitting one step in two; a trainer
step takes seconds. Override with ``DAYTONA_BATCH_GAP_SECONDS``. Spans carry
``training_step_source="derived"`` so analysis can label this as an estimate.
Explicit ids (``args.daytona_training_step`` / ``DAYTONA_TRAINING_STEP``) win.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

DEFAULT_MIN_GAP_SECONDS = 1.0


def _gap_from_env() -> float:
    raw = os.environ.get("DAYTONA_BATCH_GAP_SECONDS")
    try:
        return float(raw) if raw and raw.strip() else DEFAULT_MIN_GAP_SECONDS
    except ValueError:
        return DEFAULT_MIN_GAP_SECONDS


class RolloutBatchTracker:
    """Process-local in-flight counter; thread-safe and O(1) per rollout."""

    def __init__(
        self,
        *,
        min_gap_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._min_gap = _gap_from_env() if min_gap_seconds is None else min_gap_seconds
        self._in_flight = 0
        self._step = -1
        self._idle_since: float | None = None

    def enter(self) -> int:
        with self._lock:
            if self._in_flight == 0:
                idle_for = (
                    None if self._idle_since is None else self._clock() - self._idle_since
                )
                if self._step < 0 or (idle_for is not None and idle_for >= self._min_gap):
                    self._step += 1
            self._in_flight += 1
            return self._step

    def exit(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if self._in_flight == 0:
                self._idle_since = self._clock()

    @contextmanager
    def track(self) -> Iterator[int]:
        step = self.enter()
        try:
            yield step
        finally:
            self.exit()


_DEFAULT = RolloutBatchTracker()


def default_tracker() -> RolloutBatchTracker:
    return _DEFAULT
