"""Optional RunPod helper: resolve a pod id → ``SshWorker`` (public IP + TCP port).

Uses the [RunPod REST API](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
so partners do not copy IP/port from the Connect tab by hand.

Requires:
  - ``RUNPOD_API_KEY`` (Bearer)
  - Pod created with SSH + ``22/tcp`` exposed (direct SSH, not only the proxy)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daytona_gym.gym.worker import SshWorker
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

_API = "https://rest.runpod.io/v1"


@dataclass(frozen=True)
class RunPodSshInfo:
    pod_id: str
    public_ip: str
    ssh_port: int
    user: str = "root"

    def to_worker(
        self,
        *,
        identity: str | Path | None = None,
        remote_repo: str = "/root/daytona-training-gym-spec",
        pull: bool = True,
    ) -> SshWorker:
        return SshWorker(
            host=f"{self.user}@{self.public_ip}",
            port=self.ssh_port,
            identity=identity
            or os.environ.get("DAYTONA_GYM_SSH_IDENTITY")
            or os.environ.get("RUNPOD_SSH_IDENTITY"),
            remote_repo=remote_repo,
            pull=pull,
            name=f"runpod:{self.pod_id}",
        )


def resolve_runpod_ssh(
    pod_id: str | None = None,
    *,
    api_key: str | None = None,
    user: str = "root",
) -> RunPodSshInfo:
    """Fetch public IP + mapped port 22 for a running Pod."""
    pid = (pod_id or os.environ.get("RUNPOD_POD_ID") or "").strip()
    if not pid:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "pod_id required (or set RUNPOD_POD_ID)",
        )
    key = (api_key or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "RUNPOD_API_KEY required to resolve pod SSH endpoints",
        )

    data = _get_pod(pid, api_key=key)
    info = parse_runpod_ssh(data, user=user)
    if info is None:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"pod {pid} has no direct SSH yet — wait until RUNNING with 22/tcp "
            "exposed (Connect → SSH over exposed TCP). Proxy-only "
            "ssh.runpod.io cannot do SCP.",
        )
    return info


def runpod_worker(
    pod_id: str | None = None,
    *,
    api_key: str | None = None,
    identity: str | Path | None = None,
    remote_repo: str = "/root/daytona-training-gym-spec",
    pull: bool = True,
    user: str = "root",
) -> SshWorker:
    """One-liner: ``TrainConfig(...).launch(worker=runpod_worker(\"abc123\"))``."""
    return resolve_runpod_ssh(pod_id, api_key=api_key, user=user).to_worker(
        identity=identity,
        remote_repo=remote_repo,
        pull=pull,
    )


def parse_runpod_ssh(data: dict[str, Any], *, user: str = "root") -> RunPodSshInfo | None:
    """Extract direct SSH endpoint from a Pod JSON document."""
    pod_id = str(data.get("id") or data.get("podId") or "")
    public_ip = data.get("publicIp") or data.get("public_ip")
    port: int | None = None

    mappings = data.get("portMappings") or data.get("port_mappings") or {}
    if isinstance(mappings, dict):
        raw = mappings.get("22") or mappings.get(22)
        if raw is not None:
            port = int(raw)

    ssh = data.get("ssh") or {}
    if isinstance(ssh, dict):
        direct = ssh.get("direct") or {}
        if isinstance(direct, dict):
            public_ip = public_ip or direct.get("ip") or direct.get("host")
            if direct.get("port") is not None:
                port = int(direct["port"])

    runtime = data.get("runtime") or {}
    ports = runtime.get("ports") if isinstance(runtime, dict) else None
    if isinstance(ports, list):
        for entry in ports:
            if not isinstance(entry, dict):
                continue
            if int(entry.get("private") or 0) != 22:
                continue
            public_ip = public_ip or entry.get("ip")
            if entry.get("public") is not None:
                port = int(entry["public"])
            break

    if not pod_id or not public_ip or port is None:
        return None
    return RunPodSshInfo(pod_id=pod_id, public_ip=str(public_ip), ssh_port=port, user=user)


def _get_pod(pod_id: str, *, api_key: str) -> dict[str, Any]:
    url = f"{_API}/pods/{pod_id}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"RunPod GET /pods/{pod_id} failed: HTTP {exc.code} {detail}",
        ) from exc
    except urllib.error.URLError as exc:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"RunPod API unreachable: {exc}",
        ) from exc
    data = json.loads(body)
    if not isinstance(data, dict):
        raise DaytonaError(ErrorCode.PLATFORM_ERROR, "unexpected RunPod pod payload")
    return data
