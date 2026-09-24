"""Outbound public URL for ``dg dash`` (no inbound RunPod port config).

Uses Cloudflare Quick Tunnels (``cloudflared tunnel --url …``) so a GPU box
can dial out and print an HTTPS link openable on a laptop.

Quick-tunnel hostnames are **ephemeral**: when ``cloudflared`` exits, DNS for
``*.trycloudflare.com`` goes away (browser shows DNS_PROBE / NXDOMAIN).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)

_RELEASE_BASE = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download"
)


def cache_dir() -> Path:
    override = os.environ.get("DAYTONA_GYM_CACHE")
    if override:
        path = Path(override).expanduser()
    else:
        path = Path.home() / ".cache" / "daytona-gym"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        arch = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
        return f"darwin-{arch}"
    if system == "linux":
        arch = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
        return f"linux-{arch}"
    raise RuntimeError(f"unsupported platform for cloudflared: {system}/{machine}")


def _download_url() -> str:
    tag = _platform_tag()
    mapping = {
        "linux-amd64": "cloudflared-linux-amd64",
        "linux-arm64": "cloudflared-linux-arm64",
        "darwin-amd64": "cloudflared-darwin-amd64.tgz",
        "darwin-arm64": "cloudflared-darwin-arm64.tgz",
    }
    name = mapping.get(tag)
    if not name:
        raise RuntimeError(f"no cloudflared build for {tag}")
    return f"{_RELEASE_BASE}/{name}"


def ensure_cloudflared_binary() -> Path:
    """Return cloudflared from PATH, or download into the gym cache."""
    found = shutil.which("cloudflared")
    if found:
        return Path(found)

    dest = cache_dir() / "cloudflared"
    if dest.is_file() and os.access(dest, os.X_OK):
        return dest

    url = _download_url()
    print("downloading cloudflared (one-time) …", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "asset"
        urllib.request.urlretrieve(url, archive)  # noqa: S310 — fixed release URL
        if url.endswith(".tgz"):
            import tarfile

            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp_path)
            binary = next(tmp_path.rglob("cloudflared"))
            shutil.copy2(binary, dest)
        else:
            shutil.copy2(archive, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


class QuickTunnel:
    """Background ``cloudflared`` quick tunnel process."""

    def __init__(self, local_url: str) -> None:
        self.local_url = local_url
        self.public_url: str | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._log_path = cache_dir() / f"cloudflared_{os.getpid()}.log"
        self._watch_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, *, timeout: float = 45.0) -> str:
        binary = ensure_cloudflared_binary()
        cmd = [
            str(binary),
            "tunnel",
            "--no-autoupdate",
            "--url",
            self.local_url,
        ]
        log_f = self._log_path.open("w", encoding="utf-8")
        # Own process group so a Ray/training SIGTERM to the gym process group
        # is less likely to instantly reap cloudflared (we still stop() on purpose).
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        log_f.close()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                break
            text = ""
            try:
                text = self._log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
            match = _URL_RE.search(text)
            if match:
                self.public_url = match.group(0)
                # Confirm it stays up briefly (flaky free tunnels sometimes die ASAP).
                time.sleep(1.5)
                if self._proc.poll() is not None:
                    snippet = text[-500:].strip() or "(no cloudflared output)"
                    self.stop()
                    raise RuntimeError(
                        "cloudflared published a URL then exited immediately.\n"
                        f"{snippet}\nlog={self._log_path}"
                    )
                self._start_watchdog()
                return self.public_url
            time.sleep(0.1)

        snippet = ""
        try:
            snippet = self._log_path.read_text(encoding="utf-8", errors="replace")[
                -500:
            ].strip()
        except OSError:
            pass
        self.stop()
        raise RuntimeError(
            "cloudflared did not publish a trycloudflare.com URL in time.\n"
            f"{snippet or '(no cloudflared output)'}\nlog={self._log_path}"
        )

    def _start_watchdog(self) -> None:
        self._watch_stop.clear()

        def _watch() -> None:
            while not self._watch_stop.wait(5.0):
                if self._proc is not None and self._proc.poll() is not None:
                    code = self._proc.returncode
                    print(
                        f"cloudflared exited (code={code}) — "
                        "public URL is dead (DNS will fail). "
                        f"See {self._log_path}. Re-run launch for a new link.",
                        flush=True,
                    )
                    return

        self._watch_thread = threading.Thread(target=_watch, daemon=True)
        self._watch_thread.start()

    def stop(self) -> None:
        self._watch_stop.set()
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, 15)  # SIGTERM process group
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    self._proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._proc.pid, 9)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        self._proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
        self._proc = None


def parse_public_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    return match.group(0) if match else None
