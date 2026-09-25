"""RunPod helper: resolve a pod id → ``SshWorker`` for laptop agents.

Prefer the **proxy** SSH endpoint (``user@ssh.runpod.io``). That works for every
pod with account SSH keys — including slime images that only run
``sleep infinity`` and never start container ``sshd``.

Direct TCP (``root@PUBLIC_IP -p PORT``) is optional when sshd is actually up;
most coding-agent flows should use the proxy + PTY shell transport.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from daytona_gym.gym.worker import SshWorker
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

_API_V1 = "https://rest.runpod.io/v1"
_API_V2 = "https://api.runpod.io/v2"
_UA = "daytona-gym/0.1 (+https://github.com/danielgraviet/daytona-training-gym-spec)"


@dataclass(frozen=True)
class RunPodSshInfo:
    pod_id: str
    # Proxy (always preferred for agents)
    proxy_user: str | None = None
    proxy_host: str = "ssh.runpod.io"
    proxy_port: int = 22
    # Direct TCP (only if container sshd is listening)
    public_ip: str | None = None
    ssh_port: int | None = None
    direct_user: str = "root"

    def to_worker(
        self,
        *,
        identity: str | Path | None = None,
        remote_repo: str = "/root/daytona-training-gym-spec",
        pull: bool = True,
        prefer: Literal["proxy", "direct", "auto"] = "proxy",
    ) -> SshWorker:
        ident = (
            identity
            or os.environ.get("DAYTONA_GYM_SSH_IDENTITY")
            or os.environ.get("RUNPOD_SSH_IDENTITY")
        )
        mode = prefer
        if mode == "auto":
            mode = "direct" if self.public_ip and self.ssh_port else "proxy"

        if mode == "direct":
            if not self.public_ip or self.ssh_port is None:
                raise DaytonaError(
                    ErrorCode.PLATFORM_ERROR,
                    f"pod {self.pod_id} has no direct SSH endpoint yet",
                )
            return SshWorker(
                host=f"{self.direct_user}@{self.public_ip}",
                port=self.ssh_port,
                identity=ident,
                remote_repo=remote_repo,
                pull=pull,
                name=f"runpod-direct:{self.pod_id}",
                transport="auto",
            )

        if not self.proxy_user:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                f"pod {self.pod_id} has no proxy SSH user yet — is it RUNNING?",
            )
        return SshWorker(
            host=f"{self.proxy_user}@{self.proxy_host}",
            port=self.proxy_port,
            identity=ident,
            remote_repo=remote_repo,
            pull=pull,
            name=f"runpod:{self.pod_id}",
            transport="shell",
        )


def resolve_runpod_ssh(
    pod_id: str | None = None,
    *,
    api_key: str | None = None,
    user: str = "root",
) -> RunPodSshInfo:
    """Fetch proxy (+ optional direct) SSH endpoints for a running Pod."""
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

    data = _get_pod_v2(pid, api_key=key)
    info = parse_runpod_ssh(data, user=user)
    if info.proxy_user is None and (not info.public_ip or info.ssh_port is None):
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"pod {pid} has no SSH endpoints yet — wait until RUNNING",
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
    prefer: Literal["proxy", "direct", "auto"] = "proxy",
    git_url: str | None = None,
    create: bool = False,
    create_kwargs: dict[str, Any] | None = None,
) -> SshWorker:
    """Laptop agent default: proxy SSH + PTY shell (works without container sshd).

    Pass ``create=True`` to provision a new slime pod via the RunPod API first.
    """
    from daytona_gym.envfile import load_dotenv

    load_dotenv()
    pid = pod_id
    if create:
        info = create_pod(api_key=api_key, **(create_kwargs or {}))
        pid = info.pod_id
    w = resolve_runpod_ssh(pid, api_key=api_key, user=user).to_worker(
        identity=identity,
        remote_repo=remote_repo,
        pull=pull,
        prefer=prefer,
    )
    if git_url:
        w.git_url = git_url
    elif os.environ.get("DAYTONA_GYM_GIT_URL"):
        w.git_url = os.environ["DAYTONA_GYM_GIT_URL"]
    return w


@dataclass(frozen=True)
class CreatedPod:
    pod_id: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def create_pod(
    *,
    api_key: str | None = None,
    name: str | None = None,
    image: str = "slimerl/slime:latest",
    gpu_type_id: str | None = None,
    gpu_count: int = 1,
    cloud_type: str = "SECURE",
    volume_in_gb: int = 50,
    container_disk_in_gb: int = 50,
    ports: str = "22/tcp",
    docker_start_cmd: list[str] | None = None,
    env: dict[str, str] | None = None,
    wait_running: bool = True,
    wait_timeout_seconds: float = 600,
) -> CreatedPod:
    """Provision a RunPod GPU pod for BYO Slime training (``sleep infinity``).

    Requires ``RUNPOD_API_KEY``. Default image is ``slimerl/slime:latest``.
    SSH keys must already be configured on the RunPod account.
    """
    key = (api_key or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "RUNPOD_API_KEY required to create a pod",
        )
    gpu = (
        gpu_type_id
        or os.environ.get("RUNPOD_GPU_TYPE_ID")
        or "NVIDIA H100 80GB HBM3"
    )
    body: dict[str, Any] = {
        "name": name or f"daytona-gym-{int(time.time())}",
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": int(gpu_count),
        "cloudType": cloud_type,
        "volumeInGb": int(volume_in_gb),
        "containerDiskInGb": int(container_disk_in_gb),
        "ports": ports,
        "dockerStartCmd": docker_start_cmd or ["bash", "-lc", "sleep infinity"],
        "env": env or {},
    }
    data = _http_post(f"{_API_V1}/pods", api_key=key, body=body)
    pod_id = str(data.get("id") or data.get("podId") or "")
    if not pod_id:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"RunPod create returned no pod id: {data!r}"[:400],
        )
    if wait_running:
        _wait_pod_running(pod_id, api_key=key, timeout=wait_timeout_seconds)
    return CreatedPod(pod_id=pod_id, raw=data)


def _wait_pod_running(pod_id: str, *, api_key: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            data = _get_pod_v2(pod_id, api_key=api_key)
        except DaytonaError as exc:
            last = str(exc)
            time.sleep(5)
            continue
        status = str(
            data.get("desiredStatus")
            or data.get("status")
            or data.get("runtime", {}).get("uptimeInSeconds")
            or ""
        ).upper()
        # v2 uses desiredStatus RUNNING once up; also accept runtime presence.
        runtime = data.get("runtime") or {}
        if status == "RUNNING" or (isinstance(runtime, dict) and runtime.get("ports")):
            # Prefer having proxy SSH ready.
            info = parse_runpod_ssh(data)
            if info.proxy_user or (info.public_ip and info.ssh_port):
                return
        last = status or "unknown"
        time.sleep(5)
    raise DaytonaError(
        ErrorCode.PLATFORM_ERROR,
        f"pod {pod_id} not ready within {timeout:.0f}s (last={last})",
    )


def _http_post(url: str, *, api_key: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": _UA,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"RunPod POST {url} failed: HTTP {exc.code} {detail}",
        ) from exc
    except urllib.error.URLError as exc:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"RunPod API unreachable: {exc}",
        ) from exc
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise DaytonaError(ErrorCode.PLATFORM_ERROR, "unexpected RunPod create payload")
    return data


def parse_runpod_ssh(data: dict[str, Any], *, user: str = "root") -> RunPodSshInfo:
    """Extract proxy + direct SSH endpoints from a Pod JSON document (v1 or v2)."""
    pod_id = str(data.get("id") or data.get("podId") or "")
    proxy_user: str | None = None
    proxy_host = "ssh.runpod.io"
    proxy_port = 22
    public_ip = data.get("publicIp") or data.get("public_ip")
    port: int | None = None

    mappings = data.get("portMappings") or data.get("port_mappings") or {}
    if isinstance(mappings, dict):
        raw = mappings.get("22") or mappings.get(22)
        if raw is not None:
            port = int(raw)

    ssh = data.get("ssh") or {}
    if isinstance(ssh, dict):
        proxy = ssh.get("proxy") or {}
        if isinstance(proxy, dict) and proxy.get("username"):
            proxy_user = str(proxy["username"])
            proxy_host = str(proxy.get("host") or proxy_host)
            if proxy.get("port") is not None:
                proxy_port = int(proxy["port"])
        direct = ssh.get("direct") or {}
        if isinstance(direct, dict):
            public_ip = public_ip or direct.get("ip") or direct.get("host")
            if direct.get("port") is not None:
                port = int(direct["port"])
            if direct.get("username"):
                user = str(direct["username"])

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

    return RunPodSshInfo(
        pod_id=pod_id,
        proxy_user=proxy_user,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        public_ip=str(public_ip) if public_ip else None,
        ssh_port=port,
        direct_user=user,
    )


def _get_pod_v2(pod_id: str, *, api_key: str) -> dict[str, Any]:
    """Prefer v2 (includes ``ssh.proxy``); fall back to v1 if needed."""
    try:
        return _http_get(f"{_API_V2}/pods/{pod_id}", api_key=api_key)
    except DaytonaError as v2_exc:
        try:
            return _http_get(f"{_API_V1}/pods/{pod_id}", api_key=api_key)
        except DaytonaError:
            raise v2_exc from None


def _http_get(url: str, *, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": _UA,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        hint = ""
        if exc.code == 403 and "1010" in detail:
            hint = (
                " (Cloudflare blocked the client — ensure User-Agent is sent)"
            )
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"RunPod GET {url} failed: HTTP {exc.code} {detail}{hint}",
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
