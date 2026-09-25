"""Terminal progress UI for ``TrainingRun.wait()`` / launch banners."""

from __future__ import annotations

import sys
import time
from typing import Any

_PHASE_HINT = {
    "boot": "worker boot",
    "starting": "startup",
    "waiting": "waiting for first worker update",
    "syncing": "pulling status from GPU (not dataset/model yet)",
    "dataset": "dataset materialize (HF / Harbor → JSONL)",
    "dataset_download": "downloading dataset",
    "model_download": "downloading model weights",
    "model_convert": "converting HF → Megatron checkpoint",
    "ray_start": "Ray / Slime startup",
    "preflight": "Daytona preflight",
    "sglang": "SGLang engine",
    "megatron": "Megatron init",
    "rollout": "rollouts (Daytona sandboxes)",
    "live": "training rollouts",
    "completed": "done",
    "failed": "failed",
}


def _console():
    try:
        from rich.console import Console

        return Console(stderr=False)
    except Exception:  # noqa: BLE001
        return None


def banner_open(url: str) -> None:
    """Print the dashboard URL once — quiet, not a lecture."""
    console = _console()
    if console is not None:
        console.print(f"[dim]dashboard[/dim]  [cyan underline]{url}[/cyan underline]")
        return
    print(f"dashboard  {url}", flush=True)


def banner_complete(*, ok: bool, run_id: str, returncode: int | None, url: str | None) -> None:
    console = _console()
    if console is not None:
        console.print()
        if ok:
            console.print(f"[green]✓ Training complete[/green]  {run_id}")
        else:
            rc = f" (exit={returncode})" if returncode is not None else ""
            console.print(f"[red]✗ Training failed[/red]  {run_id}{rc}")
        if url:
            console.print(f"[dim]dashboard[/dim]  [cyan underline]{url}[/cyan underline]")
        console.print()
        return
    print(flush=True)
    if ok:
        print(f"✓ Training complete: {run_id}", flush=True)
    else:
        rc = f" (exit={returncode})" if returncode is not None else ""
        print(f"✗ Training failed: {run_id}{rc}", flush=True)
    if url:
        print(f"dashboard  {url}", flush=True)
    print(flush=True)


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _phase_label(phase: str, message: str, *, log_tail: list[str] | None = None) -> str:
    hint = _PHASE_HINT.get(phase)
    base = message
    if hint and hint.lower() not in message.lower():
        base = f"{message}  ·  {hint}"
    if log_tail:
        last = str(log_tail[-1]).strip()
        if last and last not in base:
            # Keep status line readable.
            clipped = last if len(last) <= 90 else last[:87] + "…"
            base = f"{base}  │  {clipped}"
    return base


class WaitProgress:
    """Single in-place status line (Rich Status), no spam."""

    def __init__(self, run_id: str, *, started_at: float | None = None) -> None:
        self.run_id = run_id
        self.started_at = started_at or time.time()
        self._last_key: str | None = None
        self._console = _console()
        self._status = None
        self._last_phase = "waiting"
        self._last_message = "Waiting for worker…"
        self._last_log: str | None = None
        if self._console is not None:
            try:
                from rich.status import Status

                self._status = Status("", console=self._console, spinner="dots")
                self._status.start()
            except Exception:  # noqa: BLE001
                self._status = None

    def update(self, snap: dict[str, Any] | None) -> None:
        log_tail: list[str] | None = None
        if snap:
            phase = str(snap.get("phase") or self._last_phase)
            message = str(
                snap.get("message") or snap.get("detail") or phase or self._last_message
            )
            raw_tail = snap.get("log_tail")
            if isinstance(raw_tail, list):
                log_tail = [str(x) for x in raw_tail if str(x).strip()]
            self._last_phase = phase
            self._last_message = message
        else:
            phase = self._last_phase
            message = self._last_message

        last_log = log_tail[-1] if log_tail else ""
        key = f"{phase}|{message}|{last_log}"
        now = time.time()
        elapsed = format_elapsed(now - self.started_at)
        label = _phase_label(phase, message, log_tail=log_tail)
        short_id = self.run_id[-8:] if len(self.run_id) > 8 else self.run_id
        text = f"[{elapsed}] {label}  ({short_id})"

        # Permanent line only when phase changes (not every log flicker).
        phase_only = phase
        prev_phase = (
            None if self._last_key is None else self._last_key.split("|", 1)[0]
        )
        if (
            phase_only != prev_phase
            and prev_phase is not None
            and self._console is not None
            and prev_phase not in {"waiting", "syncing"}
        ):
            # Keep a quiet breadcrumb for real worker phases only.
            self._console.print(
                f"[dim]·[/dim] [{prev_phase}] {self._last_message}",
                highlight=False,
            )
        self._last_key = f"{phase}|{message}"
        self._last_message = message

        if self._status is not None:
            self._status.update(text)
            return

        sys.stdout.write(f"\r{text}\033[K")
        sys.stdout.flush()

    def finish(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        elif self._console is None:
            sys.stdout.write("\n")
            sys.stdout.flush()
