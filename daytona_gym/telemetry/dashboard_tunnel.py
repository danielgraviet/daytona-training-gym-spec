"""Outbound public URL for ``dg dash`` (no inbound RunPod port config).

Uses Cloudflare Quick Tunnels (``cloudflared tunnel --url …``) so a GPU box
can dial out and print an HTTPS link openable on a laptop.
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

    def start(self, *, timeout: float = 45.0) -> str:
        binary = ensure_cloudflared_binary()
        cmd = [
            str(binary),
            "tunnel",
            "--no-autoupdate",
            "--url",
            self.local_url,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        deadline = time.time() + timeout
        lines: list[str] = []

        def _reader() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                lines.append(line)
                match = _URL_RE.search(line)
                if match and self.public_url is None:
                    self.public_url = match.group(0)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        while time.time() < deadline:
            if self.public_url:
                return self.public_url
            if self._proc.poll() is not None:
                break
            time.sleep(0.1)
        snippet = "".join(lines[-20:]).strip() or "(no cloudflared output)"
        self.stop()
        raise RuntimeError(
            "cloudflared did not publish a trycloudflare.com URL in time.\n"
            f"{snippet}"
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def parse_public_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    return match.group(0) if match else None
