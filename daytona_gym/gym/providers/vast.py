"""Vast.ai helper: search offers → create instance → ``SshWorker``.

Second (non-RunPod) BYO provider for Phase 3.4. Prefer direct SSH on the
instance (Vast exposes ``ssh_host`` / ``ssh_port``). The published worker image
``dtgraviet/daytona-gym-worker`` (or GHCR) is the default launch image.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daytona_gym.gym.worker import SshWorker
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

_API = "https://console.vast.ai/api/v0"
_UA = "daytona-gym/0.1 (+https://github.com/danielgraviet/daytona-training-gym-spec)"
DEFAULT_IMAGE = "docker.io/dtgraviet/daytona-gym-worker:latest"
# PyTorch/CUDA base needs host driver CUDA ≥ 12.9 (see docker/worker/Dockerfile).
DEFAULT_DISK_GB = 80


@dataclass(frozen=True)
class VastSshInfo:
    instance_id: int
    host: str
    port: int
    user: str = "root"
    gpu_name: str | None = None
    status: str | None = None

    def to_worker(
        self,
        *,
        identity: str | Path | None = None,
        remote_repo: str = "/root/daytona-training-gym-spec",
        pull: bool = True,
    ) -> SshWorker:
        ident = (
            identity
            or os.environ.get("DAYTONA_GYM_SSH_IDENTITY")
            or os.environ.get("VAST_SSH_IDENTITY")
        )
        return SshWorker(
            host=f"{self.user}@{self.host}",
            port=self.port,
            identity=ident,
            remote_repo=remote_repo,
            pull=pull,
            name=f"vast:{self.instance_id}",
            transport="auto",
        )


@dataclass(frozen=True)
class CreatedInstance:
    instance_id: int
    offer_id: int | None = None


def _api_key(explicit: str | None = None) -> str:
    key = (explicit or os.environ.get("VAST_API_KEY") or "").strip()
    if key:
        return key
    path = Path("~/.vast_api_key").expanduser()
    if path.is_file():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise DaytonaError(
        ErrorCode.USER_CODE_ERROR,
        "VAST_API_KEY required (or ~/.vast_api_key). "
        "Create one at https://cloud.vast.ai/manage-keys/",
    )


def _http(
    method: str,
    path: str,
    *,
    api_key: str,
    body: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> Any:
    url = f"{_API}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", "replace")[:400]
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"Vast API {method} {path} → HTTP {exc.code}: {err_body}",
        ) from exc
    except urllib.error.URLError as exc:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR, f"Vast API unreachable: {exc}"
        ) from exc
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR, f"Vast API returned non-JSON: {raw[:200]!r}"
        ) from exc


def search_offers(
    *,
    api_key: str | None = None,
    gpu_ram_gb: float = 24.0,
    cpu_ram_gb: float = 32.0,
    num_gpus: int = 1,
    limit: int = 20,
    verified: bool = True,
    rentable: bool = True,
    order: str = "dph_total",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search on-demand offers. ``gpu_ram`` / ``cpu_ram`` filters are in GB here."""
    key = _api_key(api_key)
    # REST API uses MB for gpu_ram / cpu_ram (CLI converts from GB).
    q: dict[str, Any] = {
        "verified": {"eq": verified},
        "rentable": {"eq": rentable},
        "rented": {"eq": False},
        "num_gpus": {"eq": num_gpus},
        "gpu_ram": {"gte": int(gpu_ram_gb * 1024)},
        "cpu_ram": {"gte": int(cpu_ram_gb * 1024)},
        "cuda_max_good": {"gte": 12.9},
        "type": "on-demand",
        "order": [[order, "asc"]],
        "limit": limit,
    }
    if extra:
        q.update(extra)
    data = _http("POST", "/bundles/", api_key=key, body=q)
    offers = data.get("offers") if isinstance(data, dict) else data
    if not isinstance(offers, list):
        return []
    return [o for o in offers if isinstance(o, dict)]


def create_instance(
    offer_id: int,
    *,
    api_key: str | None = None,
    image: str = DEFAULT_IMAGE,
    disk_gb: int = DEFAULT_DISK_GB,
    label: str = "daytona-gym",
    onstart: str | None = None,
    env: dict[str, str] | None = None,
) -> CreatedInstance:
    """Accept an offer. Returns the new contract id (instance id)."""
    key = _api_key(api_key)
    # Keep the container alive for SSH (worker image ENTRYPOINT is python/slime).
    start = onstart or "mkdir -p /root/daytona-training-gym-spec/runs && sleep infinity"
    body: dict[str, Any] = {
        "image": image,
        "disk": disk_gb,
        "label": label,
        "onstart": start,
        "runtype": "ssh_direct",
    }
    if env:
        body["env"] = env
    data = _http("PUT", f"/asks/{int(offer_id)}/", api_key=key, body=body)
    contract = data.get("new_contract") if isinstance(data, dict) else None
    if contract is None:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"Vast create did not return new_contract: {data!r}",
        )
    return CreatedInstance(instance_id=int(contract), offer_id=int(offer_id))


def destroy_instance(instance_id: int, *, api_key: str | None = None) -> None:
    key = _api_key(api_key)
    _http("DELETE", f"/instances/{int(instance_id)}/", api_key=key)


def list_instances(*, api_key: str | None = None) -> list[dict[str, Any]]:
    key = _api_key(api_key)
    data = _http("GET", "/instances/", api_key=key, query={"owner": "me"})
    instances = data.get("instances") if isinstance(data, dict) else data
    if not isinstance(instances, list):
        return []
    return [i for i in instances if isinstance(i, dict)]


def parse_vast_ssh(instance: dict[str, Any]) -> VastSshInfo:
    iid = int(instance.get("id") or instance.get("instance_id") or 0)
    host = (
        instance.get("ssh_host")
        or instance.get("public_ipaddr")
        or instance.get("host_ip")
    )
    port_raw = instance.get("ssh_port") or instance.get("ports", {}).get("22/tcp")
    if isinstance(port_raw, list) and port_raw:
        port_raw = port_raw[0].get("HostPort") if isinstance(port_raw[0], dict) else port_raw[0]
    port = int(port_raw) if port_raw else 0
    user = str(instance.get("ssh_user") or "root")
    status = str(
        instance.get("actual_status")
        or instance.get("cur_state")
        or instance.get("status_msg")
        or ""
    )
    gpu = instance.get("gpu_name") or instance.get("gpu_nam")
    if not host or not port:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"Vast instance {iid} has no SSH endpoint yet (status={status!r})",
        )
    return VastSshInfo(
        instance_id=iid,
        host=str(host),
        port=port,
        user=user,
        gpu_name=str(gpu) if gpu else None,
        status=status or None,
    )


def resolve_vast_ssh(
    instance_id: int | None = None,
    *,
    api_key: str | None = None,
) -> VastSshInfo:
    iid = instance_id
    if iid is None:
        raw = (os.environ.get("VAST_INSTANCE_ID") or "").strip()
        if not raw:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "instance_id required (or set VAST_INSTANCE_ID)",
            )
        iid = int(raw)
    key = _api_key(api_key)
    data = _http("GET", f"/instances/{int(iid)}/", api_key=key)
    # Some responses nest under "instances": [one]
    if isinstance(data, dict) and "instances" in data:
        items = data["instances"]
        if isinstance(items, list) and items:
            data = items[0]
    if not isinstance(data, dict):
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR, f"unexpected Vast instance payload: {data!r}"
        )
    # Ensure id is present
    data.setdefault("id", iid)
    return parse_vast_ssh(data)


def wait_until_ssh_ready(
    instance_id: int,
    *,
    api_key: str | None = None,
    timeout: float = 600.0,
    poll: float = 5.0,
) -> VastSshInfo:
    """Poll until SSH endpoints exist and status looks running."""
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            info = resolve_vast_ssh(instance_id, api_key=api_key)
            status = (info.status or "").lower()
            if status in {"exited", "unknown", "offline", "error"}:
                raise DaytonaError(
                    ErrorCode.PLATFORM_ERROR,
                    f"Vast instance {instance_id} will not reach running (status={status})",
                )
            if status in {"running", "connected", ""} or info.host:
                # Prefer an explicit running signal when present.
                if status in {"running", "connected"} or status == "":
                    return info
                if "run" in status:
                    return info
        except DaytonaError as exc:
            last_err = exc
            if "will not reach running" in str(exc):
                raise
        time.sleep(poll)
    raise DaytonaError(
        ErrorCode.PLATFORM_ERROR,
        f"timed out waiting for Vast SSH on {instance_id}: {last_err}",
    )


def create_and_wait(
    *,
    api_key: str | None = None,
    gpu_ram_gb: float = 24.0,
    cpu_ram_gb: float = 32.0,
    image: str = DEFAULT_IMAGE,
    disk_gb: int = DEFAULT_DISK_GB,
    label: str = "daytona-gym",
    timeout: float = 600.0,
) -> VastSshInfo:
    """Search → rent cheapest matching offer → wait for SSH."""
    offers = search_offers(
        api_key=api_key, gpu_ram_gb=gpu_ram_gb, cpu_ram_gb=cpu_ram_gb
    )
    if not offers:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            f"no Vast offers with ≥{gpu_ram_gb}GB VRAM and ≥{cpu_ram_gb}GB RAM",
        )
    offer = min(offers, key=lambda o: float(o.get("dph_total") or 1e9))
    created = create_instance(
        int(offer["id"]),
        api_key=api_key,
        image=image,
        disk_gb=disk_gb,
        label=label,
    )
    return wait_until_ssh_ready(created.instance_id, api_key=api_key, timeout=timeout)


def vast_worker(
    instance_id: int | None = None,
    *,
    api_key: str | None = None,
    identity: str | Path | None = None,
    remote_repo: str = "/root/daytona-training-gym-spec",
    pull: bool = True,
    create: bool = False,
    create_kwargs: dict[str, Any] | None = None,
) -> SshWorker:
    """Resolve (or create) a Vast instance into an ``SshWorker``."""
    from daytona_gym.envfile import load_dotenv

    load_dotenv()
    if create:
        info = create_and_wait(api_key=api_key, **(create_kwargs or {}))
    else:
        info = resolve_vast_ssh(instance_id, api_key=api_key)
    return info.to_worker(
        identity=identity, remote_repo=remote_repo, pull=pull
    )
