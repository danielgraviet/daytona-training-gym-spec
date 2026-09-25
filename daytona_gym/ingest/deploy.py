"""``dg ingest deploy --daytona`` — run the ingest service in a long-lived Daytona sandbox.

Idempotent: finds the sandbox by label (creates it the first time), starts it
if stopped, installs/updates the gym package, makes sure ``dg ingest`` is
running, checks ``/healthz`` through the preview URL, and prints the
``DAYTONA_GYM_INGEST_URL`` / ``_TOKEN`` to export. Re-run it any time the
sandbox was restarted — processes do not survive a stop/start.

Platform facts this relies on (Daytona SDK 0.210):

* Preview traffic does **not** count as sandbox activity, so auto-stop must be
  disabled or the box stops mid-training while shippers are pushing.
* Data lives on the sandbox disk (persists across stop/start). Volumes are
  object-storage backed and a poor fit for append + atomic-rename writes.
* ``public=True``: the preview proxy lets requests through and ``dg ingest``
  enforces its own token on every read and write.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from daytona_gym.ingest.config import INGEST_TOKEN_ENV, INGEST_URL_ENV

LABEL_KEY = "app"
LABEL_VALUE = "daytona-gym-ingest"
PORT = 8080
DATA_DIR = "/home/daytona/gym-data"
LOG_PATH = "/home/daytona/ingest.log"
SESSION = "dg-ingest"
REPO_URL = "https://github.com/danielgraviet/daytona-training-gym-spec.git"
DEFAULT_REF = "main"


def _log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


async def find_sandbox(daytona, name: str):
    from daytona import ListSandboxesQuery

    # Consume the whole listing: returning mid-``async for`` closes the SDK's
    # generator in another context and its OpenTelemetry hook raises.
    matches = [
        sandbox
        async for sandbox in daytona.list(ListSandboxesQuery(labels={LABEL_KEY: LABEL_VALUE}))
        if getattr(sandbox, "name", None) == name
    ]
    return matches[0] if matches else None


async def create_sandbox(daytona, name: str, token: str):
    from daytona import CreateSandboxFromSnapshotParams

    params = CreateSandboxFromSnapshotParams(
        name=name,
        labels={LABEL_KEY: LABEL_VALUE},
        public=True,
        auto_stop_interval=0,  # preview traffic isn't "activity" — never auto-stop
        auto_delete_interval=-1,  # never auto-delete when stopped
        env_vars={INGEST_TOKEN_ENV: token},
    )
    return await daytona.create(params, timeout=180)


async def read_sandbox_token(sandbox) -> str | None:
    resp = await sandbox.process.exec(f"printenv {INGEST_TOKEN_ENV}", timeout=30)
    value = (getattr(resp, "result", "") or "").strip()
    return value or None


def resolve_spec(ref: str) -> str:
    """Pin a branch/tag to its commit so re-deploys really update the code.

    pip treats an already-installed ``git+…@main`` as satisfied (same name +
    version), so redeploying a moving branch silently kept old code.
    """
    import re
    import subprocess

    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return f"git+{REPO_URL}@{ref}"
    try:
        out = subprocess.run(
            ["git", "ls-remote", REPO_URL, ref], capture_output=True, text=True, timeout=30
        ).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        out = []
    sha = out[0] if out and re.fullmatch(r"[0-9a-f]{40}", out[0]) else None
    if sha is None:
        _log(f"could not resolve {ref!r} to a commit; installing the ref directly")
        return f"git+{REPO_URL}@{ref}"
    _log(f"{ref} → {sha[:12]}")
    return f"git+{REPO_URL}@{sha}"


async def install_package(sandbox, spec: str) -> None:
    quoted = shlex.quote(spec)
    # Deps first (no-op when present), then force the gym itself to this commit.
    cmd = (
        f"python -m pip install --quiet {quoted} && "
        f"python -m pip install --quiet --force-reinstall --no-deps {quoted}"
    )
    _log(f"installing {spec} (first time can take a minute)")
    resp = await sandbox.process.exec(cmd, timeout=600)
    if getattr(resp, "exit_code", 1) != 0:
        raise RuntimeError(f"pip install failed: {(resp.result or '')[-800:]}")


async def process_healthy(sandbox) -> bool:
    resp = await sandbox.process.exec(
        f"python -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:{PORT}/healthz',timeout=3)\"",
        timeout=30,
    )
    return getattr(resp, "exit_code", 1) == 0


async def stop_server(sandbox) -> None:
    await sandbox.process.exec("pkill -f 'daytona_gym.cli ingest' || true", timeout=30)
    for _ in range(10):
        if not await process_healthy(sandbox):
            return
        await asyncio.sleep(0.5)


async def start_server(sandbox) -> None:
    from daytona import SessionExecuteRequest

    # A fresh session per start: once the previous server is killed its
    # session has exited, and Daytona refuses new commands in an exited session.
    session = f"{SESSION}-{int(time.time())}"
    await sandbox.process.create_session(session)
    command = (
        f"mkdir -p {DATA_DIR} && exec python -m daytona_gym.cli ingest "
        f"--host 0.0.0.0 --port {PORT} --data-dir {DATA_DIR} >> {LOG_PATH} 2>&1"
    )
    await sandbox.process.execute_session_command(
        session, SessionExecuteRequest(command=command, run_async=True)
    )


def wait_public_health(url: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=10) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(2)
    raise RuntimeError(f"ingest not healthy via preview URL {url}: {last}")


async def deploy(*, name: str, spec: str, token: str | None) -> tuple[str, str, str]:
    """Returns (sandbox_id, url, token)."""
    from daytona import AsyncDaytona

    async with AsyncDaytona() as daytona:
        sandbox = await find_sandbox(daytona, name)
        if sandbox is None:
            token = token or secrets.token_urlsafe(32)
            _log(f"creating sandbox {name!r} (auto-stop off, auto-delete off, public preview)")
            sandbox = await create_sandbox(daytona, name, token)
        else:
            _log(f"reusing sandbox {name!r} ({sandbox.id}), state={sandbox.state}")
            if str(sandbox.state).lower().endswith(("stopped", "archived")):
                _log("starting sandbox")
                await sandbox.start(timeout=180)
            # Re-assert lifecycle settings in case they were changed in the UI.
            await sandbox.set_autostop_interval(0)
            existing = await read_sandbox_token(sandbox)
            if token and existing and token != existing:
                raise RuntimeError(
                    f"local {INGEST_TOKEN_ENV} differs from the sandbox's; unset it to reuse the sandbox token"
                )
            token = existing or token
            if not token:
                raise RuntimeError(f"sandbox has no {INGEST_TOKEN_ENV}; recreate it")

        await install_package(sandbox, spec)
        if await process_healthy(sandbox):
            # Freshly installed code only takes effect in a new process.
            _log("restarting dg ingest on the new code (shippers retry through the gap)")
            await stop_server(sandbox)
        else:
            _log("starting dg ingest")
        await start_server(sandbox)
        for _ in range(30):
            if await process_healthy(sandbox):
                break
            await asyncio.sleep(1)
        else:
            tail = await sandbox.process.exec(f"tail -n 40 {LOG_PATH}", timeout=30)
            raise RuntimeError(f"dg ingest did not start:\n{tail.result}")

        preview = await sandbox.get_preview_link(PORT)
        url = preview.url.rstrip("/")
        _log(f"checking {url}/healthz through the preview proxy")
        wait_public_health(url)
        return sandbox.id, url, token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dg ingest deploy",
        description="Run `dg ingest` in a long-lived Daytona sandbox (idempotent).",
    )
    parser.add_argument("--daytona", action="store_true", required=True, help="Deploy to a Daytona sandbox")
    parser.add_argument("--name", default="daytona-gym-ingest", help="Sandbox name (default: daytona-gym-ingest)")
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="Git branch/tag/sha of the gym to run (default: main, pinned to its current commit)",
    )
    parser.add_argument(
        "--write-env",
        type=Path,
        default=None,
        help="Write the export lines to this file (mode 0600) instead of printing the token",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("DAYTONA_API_KEY"):
        print("DAYTONA_API_KEY is required", file=sys.stderr)
        return 2
    token = (os.environ.get(INGEST_TOKEN_ENV) or "").strip() or None
    try:
        spec = resolve_spec(args.ref)
        sandbox_id, url, token = asyncio.run(deploy(name=args.name, spec=spec, token=token))
    except Exception as exc:  # noqa: BLE001
        print(f"deploy failed: {exc}", file=sys.stderr)
        return 1

    lines = f"export {INGEST_URL_ENV}={url}\nexport {INGEST_TOKEN_ENV}={token}\n"
    print(f"\n✓ dg ingest is up  sandbox={sandbox_id}\n  dashboard: {url}/", flush=True)
    if args.write_env:
        fd = os.open(args.write_env, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(lines)
        print(f"  env written to {args.write_env} (0600) — `source` it before launching", flush=True)
    else:
        print("\nExport these before launching (laptop and/or pod):\n" + lines, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
