"""BYO GPU workers — Local (on-box) and SSH (laptop → remote GPU).

Providers (RunPod, Lambda, …) are optional later; they should *emit* a Worker.
The gym core only needs a machine that can run Slime.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from daytona_gym.gym.run import TrainingRun
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

if TYPE_CHECKING:
    from daytona_gym.gym.config import TrainConfig

_RUN_RE = re.compile(r"^__DG_RUN_ID__=(.+)$")
_TELEMETRY_RE = re.compile(r"^__DG_TELEMETRY__=(.+)$")
_DASH_RE = re.compile(r"^__DG_DASHBOARD__=(.+)$")
_RC_RE = re.compile(r"^__DG_RETURNCODE__=(\d+)$")


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
    ) -> TrainingRun:
        return config._launch_local(
            dry_run=dry_run,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
            open=open,
            open_browser=open_browser,
        )


@dataclass
class SshWorker:
    """Laptop → remote GPU over OpenSSH (direct TCP SSH, not ssh.runpod.io gateway).

    Example::

        worker = SshWorker(
            host="root@64.x.x.x",
            port=12713,
            identity="~/.ssh/id_ed25519",
            remote_repo="/root/daytona-training-gym-spec",
        )
        run = TrainConfig(...).launch(worker=worker)
    """

    host: str
    port: int | None = None
    identity: str | Path | None = None
    remote_repo: str = "/root/daytona-training-gym-spec"
    pull: bool = True
    extra_ssh_args: list[str] = field(default_factory=list)
    name: str = "ssh"

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
    ) -> TrainingRun:
        if dry_run:
            # Plan is local/CPU-safe; remote not required.
            return config.build()

        payload = config_to_remote_payload(
            config,
            remote_repo=self.remote_repo,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
            open=True if open is None else open,
            open_browser=open_browser,
        )
        return self._exec_remote_job(payload)

    def _ssh_base(self) -> list[str]:
        cmd = [
            "ssh",
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

            remote_bits = [
                f"cd {shlex.quote(self.remote_repo)}",
            ]
            if self.pull:
                remote_bits.append("git pull --ff-only || true")
            remote_bits.append(
                "export DAYTONA_API_KEY=" + shlex.quote(api_key)
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
                elif m := _RC_RE.match(text):
                    returncode = int(m.group(1))
            rc = proc.wait()
            if not run_id:
                run_id = "remote_unknown"
            if not telemetry:
                telemetry = f"{self.remote_repo}/runs/{run_id}.jsonl"

            run = TrainingRun(
                run_id=run_id,
                telemetry_path=telemetry,
                command=ssh_cmd,
                env={"DAYTONA_GYM_WORKER": self.host},
                runtime_env={"worker": "ssh", "host": self.host},
                dry_run=False,
                returncode=returncode if rc == 0 else rc,
                inspect_hint=f"dg stats {telemetry}"
                + (f"  |  {dashboard}" if dashboard else "  |  dg dash"),
                dashboard_url=dashboard,
            )
            if dashboard:
                print(flush=True)
                print("=" * 60, flush=True)
                print("  → OPEN  (from worker tunnel)", flush=True)
                print(f"  {dashboard}", flush=True)
                print("=" * 60, flush=True)
                print(flush=True)
            return run


def config_to_remote_payload(
    config: TrainConfig,
    *,
    remote_repo: str,
    skip_preflight: bool,
    preflight_timeout_seconds: float,
    open: bool,
    open_browser: bool,
) -> dict[str, Any]:
    """JSON payload consumed by ``python -m daytona_gym.gym.remote_job`` on the worker."""
    compute = config.resolved_compute()
    dataset_path = _remote_dataset_path(config, remote_repo=remote_repo)
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
        "dataset": {
            "path": dataset_path,
            "input_key": config.dataset.input_key,
            "label_key": config.dataset.label_key,
        },
        "recipe": asdict(config.recipe),
        "model": model_blob,
        "compute": compute_blob,
        "telemetry_path": None,  # worker assigns under remote runs/
        "skip_preflight": skip_preflight,
        "preflight_timeout_seconds": preflight_timeout_seconds,
        "open": open,
        "open_browser": open_browser,
    }


def _remote_dataset_path(config: TrainConfig, *, remote_repo: str) -> str:
    """Prefer repo-relative dataset path so Mac absolute paths don't leak."""
    local = config.dataset.resolved_path()
    repo = Path(config.repo).expanduser().resolve() if config.repo else None
    if repo is not None:
        try:
            rel = local.relative_to(repo)
            return str(Path(remote_repo) / rel)
        except ValueError:
            pass
    # Fall back to basename under remote examples path if it looks like coding dogfood
    name = local.name
    return f"{remote_repo.rstrip('/')}/examples/coding_dogfood/prompts/{name}"
