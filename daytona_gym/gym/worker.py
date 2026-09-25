"""BYO GPU workers — Local (on-box) and SSH (laptop → remote GPU).

Providers (RunPod, Lambda, …) are optional later; they should *emit* a Worker.
The gym core only needs a machine that can run Slime.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from daytona_gym.gym.run import TrainingRun
from daytona_gym.gym.ssh_shell import PtyShell, is_runpod_proxy_host
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

if TYPE_CHECKING:
    from daytona_gym.gym.config import TrainConfig

_RUN_RE = re.compile(r"^__DG_RUN_ID__=(.+)$")
_TELEMETRY_RE = re.compile(r"^__DG_TELEMETRY__=(.+)$")
_DASH_RE = re.compile(r"^__DG_DASHBOARD__=(.+)$")
_RC_RE = re.compile(r"^__DG_RETURNCODE__=(\d+)$")
_DETACHED_RE = re.compile(r"^__DG_DETACHED__=1$")
_WORKER_LOG_RE = re.compile(r"^__DG_WORKER_LOG__=(.+)$")
_WORKER_PID_RE = re.compile(r"^detached_pid=(\d+)")

_REMOTE_ENV_FILE = "/tmp/daytona_gym_forward.env"

_FORWARD_ENV_KEYS = (
    "DAYTONA_API_KEY",
    "DAYTONA_API_URL",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "DAYTONA_GYM_INGEST_URL",
    "DAYTONA_GYM_INGEST_TOKEN",
)


def _forward_env_from_local() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in _FORWARD_ENV_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def _export_forward_env_cmd(extra: dict[str, str] | None = None) -> str:
    """Source secrets from the remote env file, then delete it.

    Values are never typed on the command line; the file is removed right after
    sourcing so secrets do not linger in ``/tmp`` (the launched job and any
    detached child inherit them via the environment).
    """
    merged = _forward_env_from_local()
    if extra:
        merged.update({k: v for k, v in extra.items() if v})
    if not merged:
        return "true"
    return (
        f"set -a && {{ [ -f {_REMOTE_ENV_FILE} ] && . {_REMOTE_ENV_FILE}; "
        f"rm -f {_REMOTE_ENV_FILE}; }}; set +a"
    )


def _install_forward_env_via_shell(shell: Any, extra: dict[str, str] | None = None) -> None:
    """Write the remote env file (mode 0600) without echoing secrets.

    Terminal echo is disabled while the (reversible) base64 chunks are typed, and
    local output is muted, so neither the PTY transcript nor our stdout sees them.
    """
    merged = _forward_env_from_local()
    if extra:
        merged.update({k: v for k, v in extra.items() if v})
    if not merged:
        shell.run(f"rm -f {_REMOTE_ENV_FILE}")
        return
    body = "".join(f"{k}={shlex.quote(v)}\n" for k, v in merged.items())
    b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
    tmp = f"{_REMOTE_ENV_FILE}.b64"
    with _quiet_shell_output(shell):
        shell.run("stty -echo")
        try:
            shell.run(f"(umask 077 && rm -f {tmp} {_REMOTE_ENV_FILE} && : > {tmp})")
            for i in range(0, len(b64), 3000):
                chunk = b64[i : i + 3000]
                shell.run(f"printf '%s' '{chunk}' >> {tmp}")
            shell.run(
                f"(umask 077 && base64 -d {tmp} > {_REMOTE_ENV_FILE}) && rm -f {tmp}"
            )
        finally:
            shell.run("stty echo")


def _local_package_root() -> Path:
    """Repo root that contains ``daytona_gym/`` on the laptop."""
    return Path(__file__).resolve().parents[2]


def _tar_filter(tarinfo: Any) -> Any | None:
    """Skip bulky / local-only paths when syncing the package to the worker."""
    name = tarinfo.name.replace("\\", "/")
    skip_bits = (
        "/__pycache__",
        "/.pytest_cache",
        "/node_modules",
        "/dashboard_frontend/node_modules",
        "/.git",
        ".pyc",
    )
    if any(bit in name for bit in skip_bits):
        return None
    # Prefer prebuilt SPA static assets; skip the frontend sources.
    if "/dashboard_frontend/" in name and "/dashboard_static/" not in name:
        # Keep package.json-less tree out — static build is enough on worker.
        if name.rstrip("/").endswith("dashboard_frontend"):
            return None
        return None
    return tarinfo


def _sync_package_via_shell(shell: Any, remote_repo: str) -> None:
    """Overwrite remote ``daytona_gym/`` with the laptop's tree (unpushed edits).

    ``git pull`` only sees GitHub; dogfood DX often lives in a dirty local tree.
    """
    import tarfile

    root = _local_package_root()
    pkg = root / "daytona_gym"
    if not pkg.is_dir():
        print(f"package sync skipped: missing {pkg}", file=sys.stderr)
        return

    print("syncing local daytona_gym/ → worker (overwrites git clone) …", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".tgz") as tmp:
        with tarfile.open(tmp.name, mode="w:gz") as archive:
            archive.add(pkg, arcname="daytona_gym", filter=_tar_filter)
        raw = Path(tmp.name).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    print(f"  package archive {len(raw) // 1024} KiB", flush=True)
    remote_tgz = "/tmp/daytona_gym_pkg.tgz"
    with _quiet_shell_output(shell):
        shell.run(f"rm -f {remote_tgz}.b64 {remote_tgz}")
        for i in range(0, len(b64), 3000):
            chunk = b64[i : i + 3000]
            shell.run(f"printf '%s' '{chunk}' >> {remote_tgz}.b64")
        shell.run(
            f"base64 -d {remote_tgz}.b64 > {remote_tgz} && rm -f {remote_tgz}.b64"
        )
        # Ensure repo exists, then replace package tree.
        shell.run(
            f"mkdir -p {shlex.quote(remote_repo)} && "
            f"rm -rf {shlex.quote(remote_repo + '/daytona_gym')} && "
            f"tar xzf {remote_tgz} -C {shlex.quote(remote_repo)} && "
            f"rm -f {remote_tgz}"
        )
    print("package sync done", flush=True)


def _sync_package_via_scp(worker: Any) -> None:
    """Same as PtyShell sync, using scp (exec transport)."""
    import tarfile

    root = _local_package_root()
    pkg = root / "daytona_gym"
    if not pkg.is_dir():
        return
    print("syncing local daytona_gym/ → worker via scp …", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "daytona_gym.tgz"
        with tarfile.open(tgz, mode="w:gz") as archive:
            archive.add(pkg, arcname="daytona_gym", filter=_tar_filter)
        remote_tgz = "/tmp/daytona_gym_pkg.tgz"
        scp = worker._scp_base() + [str(tgz), f"{worker.host}:{remote_tgz}"]
        proc = subprocess.run(scp, capture_output=True, text=True)
        if proc.returncode != 0:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                "package scp failed: " + (proc.stderr or proc.stdout or "").strip(),
            )
        cmd = (
            f"mkdir -p {shlex.quote(worker.remote_repo)} && "
            f"rm -rf {shlex.quote(worker.remote_repo + '/daytona_gym')} && "
            f"tar xzf {remote_tgz} -C {shlex.quote(worker.remote_repo)} && "
            f"rm -f {remote_tgz}"
        )
        ssh = worker._ssh_base() + [cmd]
        proc2 = subprocess.run(ssh, capture_output=True, text=True)
        if proc2.returncode != 0:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                "package extract failed: "
                + (proc2.stderr or proc2.stdout or "").strip(),
            )
    print("package sync done", flush=True)


def _start_laptop_dash_for_run(
    *,
    run_id: str,
    host: str,
    remote_repo: str,
    identity: str | Path | None,
    port: int | None = None,
    dash_port: int = 3000,
    worker_log: str | None = None,
) -> tuple[str, Any]:
    """Start localhost dash + PtyShell sync; return deep URL and handle.

    With ``DAYTONA_GYM_INGEST_URL`` set the worker ships telemetry to the
    ingest host, so the laptop just links there (no PTY sync, no local server).
    """
    from daytona_gym.gym.console_ui import banner_open
    from daytona_gym.ingest.config import IngestConfig

    ingest = IngestConfig.from_env()
    if ingest is not None:
        deep = ingest.run_url(run_id)
        banner_open(deep)
        return deep, None
    from daytona_gym.telemetry.dashboard import start_dashboard
    from daytona_gym.telemetry.dashboard_sync import (
        sync_run_live_via_pty,
        sync_runs_from_ssh,
    )
    from daytona_gym.telemetry.progress import read_progress, write_progress

    local_runs = Path("runs").resolve()
    local_runs.mkdir(parents=True, exist_ok=True)
    ident = Path(identity).expanduser() if identity else None

    if read_progress(local_runs, run_id) is None:
        write_progress(
            local_runs,
            run_id,
            phase="waiting",
            message="Waiting for worker phases (dataset → model → Ray/Slime)",
            status="running",
            detail=(
                "Laptop is polling the GPU. Real phases appear once the worker "
                "writes runs/<id>.progress.json (boot → dataset → model_* → ray_start)."
            ),
        )

    def _sync() -> None:
        try:
            # Fast path: one run's progress + worker log (what the wait UI needs).
            sync_run_live_via_pty(
                host=host,
                remote_root=remote_repo,
                run_id=run_id,
                local_runs=local_runs,
                identity=ident,
                ssh_port=port,
                worker_log=worker_log,
            )
            return
        except Exception as live_exc:  # noqa: BLE001
            live_err = live_exc
        try:
            sync_runs_from_ssh(
                target=host,
                local_runs=local_runs,
                remote_root=remote_repo,
                identity=ident,
                ssh_port=port,
            )
        except Exception as exc:  # noqa: BLE001
            # Never leave the UI on a frozen stub with no explanation.
            prev = read_progress(local_runs, run_id) or {}
            if str(prev.get("phase") or "") in {"", "waiting", "syncing"}:
                write_progress(
                    local_runs,
                    run_id,
                    phase="syncing",
                    message=f"GPU sync failed: {exc}",
                    status="running",
                    detail=f"live={live_err!s}; full={exc!s}",
                )
            print(f"dash sync: {exc}", file=sys.stderr)

    _sync()
    handle = start_dashboard(
        runs_dir=local_runs,
        port=dash_port,
        open_browser=False,
        share=False,
        quiet=True,
    )
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(3.0):
            _sync()

    thread = threading.Thread(target=_loop, name="laptop-dash-sync", daemon=True)
    thread.start()
    handle._sync_stop = stop  # type: ignore[attr-defined]
    deep = f"{handle.local_url.rstrip('/')}/run/{run_id}"
    banner_open(deep)
    return deep, handle


def _quiet_shell_output(shell: Any) -> Any:
    """Context: mute PtyShell echo during noisy package sync."""

    class _Mute:
        def __enter__(self) -> None:
            self._prev = getattr(shell, "_on_output", None)
            shell._on_output = None

        def __exit__(self, *exc: object) -> None:
            shell._on_output = self._prev

    return _Mute()


class Worker(Protocol):
    """Machine that can execute ``TrainConfig.launch``."""

    def launch(
        self,
        config: TrainConfig,
        *,
        dry_run: bool = False,
        skip_preflight: bool = False,
        preflight_timeout_seconds: float = 90,
        open: bool | None = None,
        open_browser: bool = False,
        detach: bool = False,
    ) -> TrainingRun: ...


@dataclass
class LocalWorker:
    """Run on this machine (today's proven on-box path)."""

    name: str = "local"

    def launch(
        self,
        config: TrainConfig,
        *,
        dry_run: bool = False,
        skip_preflight: bool = False,
        preflight_timeout_seconds: float = 90,
        open: bool | None = None,
        open_browser: bool = False,
        detach: bool = False,
    ) -> TrainingRun:
        return config._launch_local(
            dry_run=dry_run,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
            open=open,
            open_browser=open_browser,
            detach=detach,
        )


@dataclass
class SshWorker:
    """Laptop → remote GPU over SSH.

    ``transport``:
      - ``exec`` — classic ``scp`` + ``ssh host 'cmd'`` (needs real sshd / TCP).
      - ``shell`` — PTY console (RunPod ``ssh.runpod.io`` proxy; no container sshd).
      - ``auto`` — ``shell`` for RunPod proxy hosts; else try TCP then fall back.
    """

    host: str
    port: int | None = None
    identity: str | Path | None = None
    remote_repo: str = "/root/daytona-training-gym-spec"
    pull: bool = True
    # Clone if the worker disk was wiped (RunPod stop/start often clears /root).
    git_url: str = "https://github.com/danielgraviet/daytona-training-gym-spec.git"
    extra_ssh_args: list[str] = field(default_factory=list)
    name: str = "ssh"
    transport: Transport = "auto"

    @classmethod
    def from_env(cls) -> SshWorker:
        """Build from ``DAYTONA_GYM_SSH`` / ``_PORT`` / ``_IDENTITY`` / ``_REMOTE_REPO``."""
        host = os.environ.get("DAYTONA_GYM_SSH")
        if not host:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "DAYTONA_GYM_SSH not set (e.g. root@1.2.3.4)",
            )
        port_raw = os.environ.get("DAYTONA_GYM_SSH_PORT")
        return cls(
            host=host,
            port=int(port_raw) if port_raw else None,
            identity=os.environ.get("DAYTONA_GYM_SSH_IDENTITY"),
            remote_repo=os.environ.get(
                "DAYTONA_GYM_REMOTE_REPO", "/root/daytona-training-gym-spec"
            ),
            git_url=os.environ.get(
                "DAYTONA_GYM_GIT_URL",
                "https://github.com/danielgraviet/daytona-training-gym-spec.git",
            ),
        )

    def launch(
        self,
        config: TrainConfig,
        *,
        dry_run: bool = False,
        skip_preflight: bool = False,
        preflight_timeout_seconds: float = 90,
        open: bool | None = None,
        open_browser: bool = False,
        detach: bool = False,
    ) -> TrainingRun:
        if dry_run:
            # Plan is local/CPU-safe; remote not required.
            return config.build()

        # Detached remote runs: dash lives on the laptop (localhost). Do not open
        # a worker-side tunnel by default — markers return as soon as run_id exists.
        if open is None:
            open = not detach
        payload = config_to_remote_payload(
            config,
            remote_repo=self.remote_repo,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
            open=open,
            open_browser=open_browser,
            detach=detach,
        )
        mode = self._resolve_transport()
        if mode == "shell":
            return self._shell_remote_job(payload)
        return self._exec_remote_job(payload)

    def _resolve_transport(self) -> Literal["exec", "shell"]:
        if self.transport == "shell":
            return "shell"
        if self.transport == "exec":
            return "exec"
        if is_runpod_proxy_host(self.host):
            return "shell"
        return "exec"

    def _ensure_remote_repo_cmd(self) -> str:
        """Clone + editable install when the worker wiped ``/root`` (common on RunPod restart)."""
        repo = shlex.quote(self.remote_repo)
        url = shlex.quote(
            self.git_url
            or os.environ.get(
                "DAYTONA_GYM_GIT_URL",
                "https://github.com/danielgraviet/daytona-training-gym-spec.git",
            )
        )
        return (
            f"if [ ! -f {repo}/pyproject.toml ]; then "
            f"git clone --depth 1 {url} {repo} && pip install -e {repo}; "
            f"fi"
        )

    def _ssh_base(self, *, force_tty: bool = False) -> list[str]:
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=20",
        ]
        if force_tty:
            cmd.append("-tt")
        if self.identity is not None:
            cmd.extend(["-i", str(Path(self.identity).expanduser())])
        if self.port is not None:
            cmd.extend(["-p", str(self.port)])
        cmd.extend(self.extra_ssh_args)
        cmd.append(self.host)
        return cmd

    def _scp_base(self) -> list[str]:
        cmd = [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=20",
        ]
        if self.identity is not None:
            cmd.extend(["-i", str(Path(self.identity).expanduser())])
        if self.port is not None:
            cmd.extend(["-P", str(self.port)])
        return cmd

    def _exec_remote_job(self, payload: dict[str, Any]) -> TrainingRun:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "DAYTONA_API_KEY must be set locally to forward to the worker",
            )

        with tempfile.TemporaryDirectory() as tmp:
            local_job = Path(tmp) / "job.json"
            local_job.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            remote_job = f"/tmp/daytona_gym_job_{os.getpid()}.json"

            scp = self._scp_base() + [str(local_job), f"{self.host}:{remote_job}"]
            scp_proc = subprocess.run(scp, capture_output=True, text=True)
            if scp_proc.returncode != 0:
                err = (scp_proc.stderr or "").strip()
                hint = (
                    "scp to worker failed — use direct TCP SSH "
                    f"(root@IP -p PORT), not ssh.runpod.io.\n{err}"
                )
                low = err.lower()
                if "connection refused" in low:
                    hint = (
                        f"SSH port is mapped but nothing is listening on "
                        f"{self.host}"
                        + (f":{self.port}" if self.port else "")
                        + " (connection refused).\n"
                        "RunPod exposed 22/tcp, but sshd is not running in the "
                        "container.\n\n"
                        "Fix one of:\n"
                        "  1. Pod Connect → confirm 'SSH over exposed TCP' works\n"
                        "     (official PyTorch template + SSH Terminal Access, "
                        "or start sshd in the container start command).\n"
                        "  2. Or launch on the box instead:\n"
                        "     python examples/gym_sdk/runpod_dogfood.py --launch\n"
                        f"\n{err}"
                    )
                elif "permission denied" in low:
                    hint = (
                        "SSH auth failed — check DAYTONA_GYM_SSH_IDENTITY matches "
                        "the public key in the RunPod pod env / account SSH keys.\n"
                        f"{err}"
                    )
                raise DaytonaError(ErrorCode.PLATFORM_ERROR, hint)

            # Write secrets to a remote env file (never on the ssh command line).
            merged = _forward_env_from_local()
            if merged:
                local_env = Path(tmp) / "forward.env"
                fd = os.open(local_env, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write("".join(f"{k}={shlex.quote(v)}\n" for k, v in merged.items()))
                # Pre-create with 0600 so scp does not land a world-readable file.
                subprocess.run(
                    self._ssh_base()
                    + [f"(umask 077 && rm -f {_REMOTE_ENV_FILE} && : > {_REMOTE_ENV_FILE})"],
                    capture_output=True,
                    text=True,
                )
                scp_env = self._scp_base() + [
                    str(local_env),
                    f"{self.host}:{_REMOTE_ENV_FILE}",
                ]
                env_proc = subprocess.run(scp_env, capture_output=True, text=True)
                if env_proc.returncode != 0:
                    raise DaytonaError(
                        ErrorCode.PLATFORM_ERROR,
                        "failed to scp forward env file: "
                        + (env_proc.stderr or env_proc.stdout or "").strip(),
                    )

            # Ensure repo exists, then overlay laptop package.
            ensure = self._ssh_base() + [
                self._ensure_remote_repo_cmd()
                + (
                    f" && cd {shlex.quote(self.remote_repo)}"
                    " && (git pull --ff-only || true)"
                    if self.pull
                    else ""
                )
            ]
            subprocess.run(ensure, capture_output=True, text=True)
            _sync_package_via_scp(self)

            remote_bits = [
                f"cd {shlex.quote(self.remote_repo)}",
            ]
            remote_bits.append(_export_forward_env_cmd())
            remote_bits.append(
                f"export PYTHONPATH={shlex.quote(self.remote_repo)}"
                "${PYTHONPATH:+:$PYTHONPATH}"
            )
            remote_bits.append(
                "python -m daytona_gym.gym.remote_job " + shlex.quote(remote_job)
            )
            remote_cmd = " && ".join(remote_bits)

            ssh_cmd = self._ssh_base() + [remote_cmd]
            print(f"ssh worker {self.host}" + (f":{self.port}" if self.port else ""), flush=True)
            print(f"remote_repo={self.remote_repo}", flush=True)

            proc = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            run_id = ""
            telemetry = ""
            dashboard: str | None = None
            returncode = 1
            detached = False
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                text = line.strip()
                if m := _RUN_RE.match(text):
                    run_id = m.group(1).strip()
                elif m := _TELEMETRY_RE.match(text):
                    telemetry = m.group(1).strip()
                elif m := _DASH_RE.match(text):
                    dashboard = m.group(1).strip()
                elif _DETACHED_RE.match(text):
                    detached = True
                elif m := _RC_RE.match(text):
                    returncode = int(m.group(1))
            rc = proc.wait()
            if not run_id:
                run_id = "remote_unknown"
            if not telemetry:
                telemetry = f"{self.remote_repo}/runs/{run_id}.jsonl"

            final_rc = None if detached else (returncode if rc == 0 else rc)
            laptop_dash = None
            if detached and run_id and run_id != "remote_unknown":
                try:
                    dashboard, laptop_dash = _start_laptop_dash_for_run(
                        run_id=run_id,
                        host=self.host,
                        remote_repo=self.remote_repo,
                        identity=self.identity,
                        port=self.port,
                        dash_port=int(os.environ.get("DAYTONA_GYM_DASH_PORT", "3000")),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"laptop dash failed: {exc}", file=sys.stderr)
            run = TrainingRun(
                run_id=run_id,
                telemetry_path=str(Path("runs") / f"{run_id}.jsonl"),
                command=ssh_cmd,
                env={"DAYTONA_GYM_WORKER": self.host},
                runtime_env={
                    "worker": "ssh",
                    "host": self.host,
                    "detached": detached,
                    "remote_telemetry": telemetry,
                },
                dry_run=False,
                returncode=final_rc,
                inspect_hint=f"dg stats runs/{run_id}.jsonl"
                + (f"  |  {dashboard}" if dashboard else "  |  dg dash"),
                dashboard_url=dashboard,
                detached=detached,
                status=(
                    "running"
                    if detached
                    else ("failed" if int(final_rc or 0) != 0 else "completed")
                ),
            )
            if laptop_dash is not None:
                run._dashboard = laptop_dash
            elif dashboard:
                from daytona_gym.gym.console_ui import banner_open

                banner_open(dashboard)
            return run

    def _shell_remote_job(self, payload: dict[str, Any]) -> TrainingRun:
        """Run via interactive PTY shell (RunPod proxy / no container sshd)."""
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "DAYTONA_API_KEY must be set locally to forward to the worker",
            )

        remote_job = f"/tmp/daytona_gym_job_{os.getpid()}.json"
        b64 = base64.b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

        ssh_cmd = self._ssh_base(force_tty=True)
        print(
            f"ssh shell worker {self.host}"
            + (f":{self.port}" if self.port else "")
            + "  (PTY — RunPod proxy / no scp)",
            flush=True,
        )
        print(f"remote_repo={self.remote_repo}", flush=True)

        def on_output(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        captured = []
        interrupted = False

        def capture(text: str) -> None:
            captured.append(text)
            on_output(text)

        try:
            with PtyShell(ssh_cmd, on_output=capture, connect_timeout=60) as shell:
                print("uploading job payload…", flush=True)
                with _quiet_shell_output(shell):
                    shell.run(
                        f"rm -f {shlex.quote(remote_job)}.b64 {shlex.quote(remote_job)}"
                    )
                    for i in range(0, len(b64), 3000):
                        chunk = b64[i : i + 3000]
                        shell.run(
                            f"printf '%s' '{chunk}' >> {shlex.quote(remote_job)}.b64"
                        )
                    shell.run(
                        f"base64 -d {shlex.quote(remote_job)}.b64 > {shlex.quote(remote_job)}"
                    )
                    _install_forward_env_via_shell(shell)

                # Clone/pull first, then overlay laptop package (unpushed DX fixes).
                print("ensuring remote repo…", flush=True)
                shell.run(
                    self._ensure_remote_repo_cmd()
                    + (
                        f" && cd {shlex.quote(self.remote_repo)}"
                        " && (git pull --ff-only || true)"
                        if self.pull
                        else ""
                    )
                )
                _sync_package_via_shell(shell, self.remote_repo)

                print("starting remote training job…", flush=True)
                # Prefer the synced repo tree over a stale site-packages install.
                launch = (
                    f"cd {shlex.quote(self.remote_repo)}"
                    + " && "
                    + _export_forward_env_cmd()
                    + f" && export PYTHONPATH={shlex.quote(self.remote_repo)}"
                    + '${PYTHONPATH:+:$PYTHONPATH}'
                    + " && python -m daytona_gym.gym.remote_job "
                    + shlex.quote(remote_job)
                    + "; echo __DG_SHELL_DONE__"
                )
                # Training can run for a long time — no wall timeout.
                shell.run(
                    launch, timeout=None, wait_done_marker="__DG_SHELL_DONE__"
                )
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted — closed remote SSH session.", flush=True)

        text = "".join(captured)
        run_id = ""
        telemetry = ""
        dashboard: str | None = None
        worker_log: str | None = None
        returncode = 130 if interrupted else 1
        detached = False
        for line in text.replace("\r", "\n").splitlines():
            stripped = line.strip()
            if m := _RUN_RE.match(stripped):
                run_id = m.group(1).strip()
            elif m := _TELEMETRY_RE.match(stripped):
                telemetry = m.group(1).strip()
            elif m := _DASH_RE.match(stripped):
                dashboard = m.group(1).strip()
            elif m := _WORKER_LOG_RE.match(stripped):
                worker_log = m.group(1).strip()
            elif _DETACHED_RE.match(stripped):
                detached = True
            elif m := _RC_RE.match(stripped):
                if not interrupted:
                    returncode = int(m.group(1))

        if not run_id:
            run_id = "remote_unknown"
        if not telemetry:
            telemetry = f"{self.remote_repo}/runs/{run_id}.jsonl"
        if detached and not interrupted:
            returncode = None

        laptop_dash = None
        # Prefer localhost dash on the laptop; ignore Cloudflare from the worker.
        if detached and not interrupted and run_id and run_id != "remote_unknown":
            try:
                dashboard, laptop_dash = _start_laptop_dash_for_run(
                    run_id=run_id,
                    host=self.host,
                    remote_repo=self.remote_repo,
                    identity=self.identity,
                    port=self.port,
                    dash_port=int(os.environ.get("DAYTONA_GYM_DASH_PORT", "3000")),
                    worker_log=worker_log,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"laptop dash failed: {exc}", file=sys.stderr)

        run = TrainingRun(
            run_id=run_id,
            telemetry_path=str(Path("runs") / f"{run_id}.jsonl"),
            command=ssh_cmd,
            env={"DAYTONA_GYM_WORKER": self.host},
            runtime_env={
                "worker": "ssh-shell",
                "host": self.host,
                "detached": detached,
                "remote_telemetry": telemetry,
                "worker_log": worker_log,
            },
            dry_run=False,
            returncode=returncode,
            inspect_hint=f"dg stats runs/{run_id}.jsonl"
            + (f"  |  {dashboard}" if dashboard else "  |  dg dash"),
            dashboard_url=dashboard,
            detached=detached,
            status=(
                "running"
                if detached
                else ("failed" if int(returncode or 0) != 0 else "completed")
            ),
        )
        if laptop_dash is not None:
            run._dashboard = laptop_dash
        return run


def config_to_remote_payload(
    config: TrainConfig,
    *,
    remote_repo: str,
    skip_preflight: bool,
    preflight_timeout_seconds: float,
    open: bool,
    open_browser: bool,
    detach: bool = False,
) -> dict[str, Any]:
    """JSON payload consumed by ``python -m daytona_gym.gym.remote_job`` on the worker."""
    from daytona_gym.gym.dataset import (
        PromptJsonlDataset,
        serialize_dataset,
    )

    compute = config.resolved_compute()
    if isinstance(config.dataset, PromptJsonlDataset):
        dataset_blob = {
            "kind": "prompt_jsonl",
            "path": _remote_dataset_path(config, remote_repo=remote_repo),
            "input_key": config.dataset.input_key,
            "label_key": config.dataset.label_key,
        }
    else:
        # HF / Harbor: send compact config; worker materializes JSONL.
        dataset_blob = serialize_dataset(config.dataset)
    model_blob: dict[str, Any] | None = None
    if config.model is not None:
        model_blob = asdict(config.model)
    compute_blob = None
    if config.compute is not None or config.model is None:
        compute_blob = {
            "slime_root": str(compute.slime_root),
            "megatron_root": str(compute.megatron_root),
            "hf_checkpoint": str(compute.hf_checkpoint),
            "ref_load": str(compute.ref_load),
            "model_script": compute.model_script,
            "sglang_mem_fraction": compute.sglang_mem_fraction,
            "save_dir": str(compute.save_dir),
            "api_url": compute.api_url,
            "num_gpus": compute.num_gpus,
        }
    return {
        "run_name": config.run_name,
        "repo": remote_repo,
        "dataset": dataset_blob,
        "recipe": asdict(config.recipe),
        "model": model_blob,
        "compute": compute_blob,
        "telemetry_path": None,  # worker assigns under remote runs/
        "skip_preflight": skip_preflight,
        "preflight_timeout_seconds": preflight_timeout_seconds,
        "open": open,
        "open_browser": open_browser,
        "detach": detach,
        "gpu_cost_per_hour": config.gpu_cost_per_hour,
        # Secrets travel only via the 0600 env file, never in the job payload.
        "forward_env_keys": sorted(_forward_env_from_local()),
    }


def _remote_dataset_path(config: TrainConfig, *, remote_repo: str) -> str:
    """Prefer repo-relative dataset path so Mac absolute paths don't leak."""
    from daytona_gym.gym.dataset import PromptJsonlDataset

    assert isinstance(config.dataset, PromptJsonlDataset)
    local = config.dataset.resolved_path()
    repo = Path(config.repo).expanduser().resolve() if config.repo else None
    if repo is not None:
        try:
            rel = local.relative_to(repo)
            return str(Path(remote_repo) / rel)
        except ValueError:
            pass
    name = local.name
    return f"{remote_repo.rstrip('/')}/examples/coding_dogfood/prompts/{name}"
