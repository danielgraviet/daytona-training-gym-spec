"""Pull ``runs/`` from a GPU host so ``dg dash`` works on a laptop."""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path


def parse_remote_target(target: str) -> tuple[str, int | None]:
    """Split ``user@host:PORT`` (numeric port) from a plain ``user@host``."""
    target = target.strip()
    if not target:
        return target, None
    match = re.fullmatch(r"(.+):(\d{2,5})", target)
    if match and "@" in match.group(1):
        return match.group(1), int(match.group(2))
    return target, None


def resolve_remote_target(cli_remote: str | None) -> str | None:
    if cli_remote:
        return cli_remote
    return os.environ.get("DAYTONA_GYM_SSH") or None


def resolve_ssh_port(cli_port: int | None, target_port: int | None) -> int | None:
    if cli_port is not None:
        return cli_port
    if target_port is not None:
        return target_port
    raw = os.environ.get("DAYTONA_GYM_SSH_PORT")
    if raw:
        return int(raw)
    return None


def resolve_identity(cli_identity: Path | None) -> Path | None:
    if cli_identity is not None:
        return Path(cli_identity).expanduser()
    env_ident = os.environ.get("DAYTONA_GYM_SSH_IDENTITY")
    if env_ident:
        return Path(env_ident).expanduser()
    return None


def sync_runs_from_ssh(
    *,
    target: str,
    local_runs: Path,
    remote_root: str | None = None,
    identity: Path | None = None,
    ssh_port: int | None = None,
) -> Path:
    """Mirror remote ``<root>/runs`` into ``local_runs``.

    Strategy (RunPod-aware):
    1. ``scp -r`` — works on direct TCP SSH when that port is up.
    2. ``ssh`` + base64(tar) with stdin attached to ``/dev/tty`` — works through
       ``ssh.runpod.io``, which requires a real PTY and blocks SCP / ``-L``.

    Prefer ``user@ssh.runpod.io`` (stable) over ``root@PUBLIC_IP`` (rotates / flaky).
    """
    local_runs = Path(local_runs).resolve()
    local_runs.mkdir(parents=True, exist_ok=True)

    host, embedded_port = parse_remote_target(target)
    port = resolve_ssh_port(ssh_port, embedded_port)
    ident = resolve_identity(identity)

    root = (remote_root or os.environ.get("DAYTONA_GYM_SSH_ROOT") or "daytona-training-gym-spec").strip()
    root = root.removeprefix("~/").rstrip("/")
    remote_runs = f"{root}/runs" if not root.startswith("/") else f"{root.rstrip('/')}/runs"

    errors: list[str] = []

    scp_err = _try_scp(
        host=host,
        remote_runs=remote_runs,
        local_runs=local_runs,
        identity=ident,
        port=port,
    )
    if scp_err is None:
        return local_runs
    errors.append(f"scp: {scp_err}")

    try:
        _try_tty_tar_b64(
            host=host,
            root=root,
            local_runs=local_runs,
            identity=ident,
            port=port,
        )
        return local_runs
    except RuntimeError as exc:
        errors.append(f"ssh+tar: {exc}")

    raise RuntimeError(
        "failed to pull runs/ from {host}.\n{errs}\n\n"
        "RunPod notes:\n"
        "  • Prefer:  dg dash --remote <pod-user>@ssh.runpod.io -i ~/.ssh/<key>\n"
        "    (run that from your Mac Terminal — RunPod needs a real TTY)\n"
        "  • Avoid relying on root@PUBLIC_IP —p … (IP/port often dies)\n"
        "  • Or on the GPU box:  dg dash --export runs/dashboard.html\n"
        "    then download that one file however you can (Jupyter, web terminal, …)"
        .format(host=host, errs="\n".join(errors))
    )


def _try_scp(
    *,
    host: str,
    remote_runs: str,
    local_runs: Path,
    identity: Path | None,
    port: int | None,
) -> str | None:
    args = [
        "scp",
        "-r",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
    ]
    if identity is not None:
        args.extend(["-i", str(identity)])
    if port is not None:
        args.extend(["-P", str(port)])
    args.extend([f"{host}:{remote_runs}/.", str(local_runs)])
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode == 0:
        return None
    return (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()


def _try_tty_tar_b64(
    *,
    host: str,
    root: str,
    local_runs: Path,
    identity: Path | None,
    port: int | None,
) -> None:
    """Pull via ssh with a controlling TTY (required by ssh.runpod.io)."""
    import shlex

    if root.startswith("/"):
        tar_cd = shlex.quote(root)
        remote_body = (
            "echo __DG_BEGIN__; "
            f"tar czf - -C {tar_cd} runs | base64; "
            "echo; echo __DG_END__"
        )
    else:
        # $HOME expansion on remote — keep unquoted path segment
        remote_body = (
            "echo __DG_BEGIN__; "
            f'tar czf - -C "$HOME/{root}" runs | base64; '
            "echo; echo __DG_END__"
        )

    cmd = [
        "ssh",
        "-tt",  # force PTY — ssh.runpod.io rejects sessions without one
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
    ]
    if identity is not None:
        cmd.extend(["-i", str(identity)])
    if port is not None:
        cmd.extend(["-p", str(port)])
    cmd.extend([host, remote_body])

    tty_in = None
    try:
        tty_in = open("/dev/tty", "rb", buffering=0)  # noqa: SIM115
    except OSError as exc:
        raise RuntimeError(
            "no controlling terminal (/dev/tty). "
            "Run `dg dash --remote …` from your Mac Terminal app "
            "(Cursor's agent shell can't satisfy RunPod's PTY check)."
        ) from exc

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=tty_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        rc = proc.wait()
    finally:
        tty_in.close()

    text = stdout.decode("utf-8", "replace")
    # strip CR from PTY
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "__DG_BEGIN__" not in text or "__DG_END__" not in text:
        err = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            err
            or f"ssh exit {rc}; no transfer markers in output "
            f"(first 200 chars: {text[:200]!r})"
        )
    payload = text.split("__DG_BEGIN__", 1)[1].split("__DG_END__", 1)[0]
    # drop ssh noise lines; keep base64 alphabet
    b64 = "".join(
        line.strip()
        for line in payload.splitlines()
        if line.strip() and not line.strip().startswith("--")
    )
    # filter to base64 charset
    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", b64)
    if not b64:
        raise RuntimeError("empty archive from remote")
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"base64 decode failed: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=".tgz") as tmp:
        tmp.write(raw)
        tmp.flush()
        with tarfile.open(tmp.name, mode="r:gz") as archive:
            _safe_extract(archive, local_runs.parent)


def _safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive:
        name = member.name
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
