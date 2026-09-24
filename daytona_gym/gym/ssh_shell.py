"""PTY interactive SSH shell — required for RunPod ``ssh.runpod.io`` proxy.

The proxy ignores ``ssh host 'cmd'`` / SCP and only offers a console. Agents on
a laptop must open a PTY, wait for the prompt, then type commands.
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import subprocess
import time
from typing import Callable

from daytona_gym.runtime.errors import DaytonaError, ErrorCode

# Shell prompt at end of buffer (best-effort; completion uses markers).
_PROMPT_RE = re.compile(r"[#$] ?\Z")


class PtyShell:
    """Drive an interactive ``ssh -tt`` session via ``pty.openpty``."""

    def __init__(
        self,
        ssh_argv: list[str],
        *,
        on_output: Callable[[str], None] | None = None,
        connect_timeout: float = 45,
    ) -> None:
        self._on_output = on_output
        self._buf = ""
        master, slave = pty.openpty()
        self._master = master
        self._proc = subprocess.Popen(
            ssh_argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        try:
            self._wait_until(
                lambda: bool(_PROMPT_RE.search(self._buf[-120:])),
                timeout=connect_timeout,
                what="shell prompt",
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        try:
            os.close(self._master)
        except OSError:
            pass

    def __enter__(self) -> PtyShell:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write(self, text: str) -> None:
        os.write(self._master, text.encode("utf-8", "replace"))

    def _read_some(self, timeout: float) -> bool:
        r, _, _ = select.select([self._master], [], [], timeout)
        if self._master not in r:
            return False
        try:
            chunk = os.read(self._master, 8192)
        except OSError:
            return False
        if not chunk:
            return False
        text = chunk.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
        self._buf += text
        if self._on_output is not None:
            self._on_output(text)
        return True

    def _wait_until(
        self,
        pred: Callable[[], bool],
        *,
        timeout: float | None,
        what: str,
    ) -> None:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if pred():
                return
            if self._proc.poll() is not None:
                raise DaytonaError(
                    ErrorCode.PLATFORM_ERROR,
                    f"SSH session ended while waiting for {what} "
                    f"(exit {self._proc.returncode}). "
                    "Check DAYTONA_GYM_SSH_IDENTITY / RunPod SSH keys.",
                )
            wait = 0.5
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise DaytonaError(
                        ErrorCode.PLATFORM_ERROR,
                        f"timed out waiting for {what}. "
                        f"Last output: {self._buf[-400]!r}",
                    )
                wait = min(0.5, remaining)
            self._read_some(wait)

    def run(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        wait_done_marker: str | None = None,
    ) -> str:
        """Send ``command`` and wait until ``marker`` appears in output."""
        marker = wait_done_marker or f"__DG_E_{time.time_ns()}__"
        if wait_done_marker is None:
            # Short commands: append our own completion marker.
            line = command.rstrip("\n") + f" ; echo {marker}"
        else:
            # Caller already includes the marker (e.g. long training jobs).
            line = command.rstrip("\n")
            marker = wait_done_marker

        pre = len(self._buf)
        self._write(line + "\n")
        # Require the marker on its own line so the echoed command line does not
        # count (it contains ``echo <marker>`` as a substring).
        done = re.compile(rf"(?:^|\n){re.escape(marker)}(?:\n|$)")

        self._wait_until(
            lambda: done.search(self._buf[pre:]) is not None,
            timeout=timeout,
            what=f"marker {marker}",
        )
        return self._buf[pre:]


def is_runpod_proxy_host(host: str) -> bool:
    h = host.split("@", 1)[-1].lower()
    return h == "ssh.runpod.io" or h.endswith(".runpod.io")
