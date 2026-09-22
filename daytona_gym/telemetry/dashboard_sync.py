"""Pull ``runs/`` from a GPU host so ``dg dash`` works on a laptop."""

from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
from pathlib import Path


def sync_runs_from_ssh(
    *,
    target: str,
    local_runs: Path,
    remote_root: str | None = None,
    identity: Path | None = None,
) -> Path:
    """Mirror ``$HOME/<remote_root>/runs`` into ``local_runs`` via ``ssh`` + ``tar``.

    Uses ``tar`` over SSH instead of ``rsync`` — RunPod's SSH gateway often breaks rsync.
    """
    local_runs = Path(local_runs).resolve()
    local_runs.mkdir(parents=True, exist_ok=True)
    root = (remote_root or os.environ.get("DAYTONA_GYM_SSH_ROOT") or "daytona-training-gym-spec").strip()
    root = root.removeprefix("~/").rstrip("/")
    if root.startswith("/"):
        remote_cmd = f"tar czf - -C {shlex.quote(root)} runs"
    else:
        # relative to $HOME so ~ expands on the remote shell
        remote_cmd = f'tar czf - -C "$HOME/{root}" runs'

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    ident = identity
    if ident is None:
        env_ident = os.environ.get("DAYTONA_GYM_SSH_IDENTITY")
        if env_ident:
            ident = Path(env_ident).expanduser()
    if ident is not None:
        cmd.extend(["-i", str(Path(ident).expanduser())])
    cmd.extend([target, remote_cmd])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as archive:
            # members are typically runs/*.jsonl — extract into local_runs.parent
            _safe_extract(archive, local_runs.parent)
    except tarfile.ReadError as exc:
        stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
        proc.wait()
        raise RuntimeError(
            f"failed to read remote runs archive from {target}: {exc}\n{stderr.strip()}"
        ) from exc
    stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"ssh sync failed (exit {rc}) for {target}:\n{stderr.strip() or '(no stderr)'}"
        )
    return local_runs


def _safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive:
        name = member.name
        # expect runs/... — strip leading ./ 
        while name.startswith("./"):
            name = name[2:]
        if name in {"", "."}:
            continue
        if name.startswith("/") or ".." in Path(name).parts:
            raise RuntimeError(f"refusing unsafe tar member: {member.name}")
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"refusing path escape: {member.name}")
        try:
            archive.extract(member, path=dest, filter="data")
        except TypeError:
            archive.extract(member, path=dest)



def resolve_remote_target(cli_remote: str | None) -> str | None:
    if cli_remote:
        return cli_remote
    return os.environ.get("DAYTONA_GYM_SSH") or None
