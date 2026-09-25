"""Pull ``runs/`` from a GPU host so ``dg dash`` works on a laptop."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _slice_marked_block(text: str, begin: str, end: str) -> str:
    """Return text between markers that appear on their own lines.

    PtyShell echoes the full command line before it runs. That echo contains
    the marker *strings* mid-line (``echo __DG_BEGIN__; …``), so naive
    ``str.split(marker)`` grabs an empty slice inside the echo and drops the
    real payload. Line-anchored markers match only the printed delimiters.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    begin_re = re.compile(rf"(?:^|\n){re.escape(begin)}(?:\n|$)")
    end_re = re.compile(rf"(?:^|\n){re.escape(end)}(?:\n|$)")
    m_begin = begin_re.search(text)
    if m_begin is None:
        raise RuntimeError(f"missing begin marker {begin!r}: {text[-240]!r}")
    m_end = end_re.search(text, m_begin.end())
    if m_end is None:
        raise RuntimeError(f"missing end marker {end!r}: {text[-240]!r}")
    return text[m_begin.end() : m_end.start()]


def _b64_section_after_label(
    src: str, label: str, next_label: str | None = None
) -> str:
    """Extract base64 after a line-anchored label (ignore mid-line command echo)."""
    label_re = re.compile(rf"(?:^|\n){re.escape(label)}(?:\n|$)")
    m = label_re.search(src)
    if m is None:
        return ""
    chunk = src[m.end() :]
    if next_label:
        next_re = re.compile(rf"(?:^|\n){re.escape(next_label)}(?:\n|$)")
        m_next = next_re.search(chunk)
        if m_next is not None:
            chunk = chunk[: m_next.start()]
    return re.sub(
        r"[^A-Za-z0-9+/=]",
        "",
        "".join(
            ln.strip()
            for ln in chunk.splitlines()
            if ln.strip()
            and not ln.strip().startswith("root@")
            and "base64" not in ln
            and "echo " not in ln
            and "tail " not in ln
            and "printf " not in ln
        ),
    )


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


def resolve_auto_remote() -> tuple[str, Path | None, str] | None:
    """If RunPod/.env is set, return ``(user@host, identity, remote_root)``."""
    from daytona_gym.envfile import load_dotenv

    load_dotenv()
    pod = (os.environ.get("RUNPOD_POD_ID") or "").strip()
    key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if pod and key:
        try:
            from daytona_gym.gym.providers.runpod import resolve_runpod_ssh

            info = resolve_runpod_ssh(pod, api_key=key)
        except Exception:
            info = None
        if info is not None and info.proxy_user:
            host = f"{info.proxy_user}@{info.proxy_host}"
            ident = resolve_identity(None)
            root = (
                os.environ.get("DAYTONA_GYM_REMOTE_REPO")
                or os.environ.get("DAYTONA_GYM_SSH_ROOT")
                or "/root/daytona-training-gym-spec"
            )
            return host, ident, root

    target = resolve_remote_target(None)
    if not target:
        return None
    ident = resolve_identity(None)
    root = (
        os.environ.get("DAYTONA_GYM_REMOTE_REPO")
        or os.environ.get("DAYTONA_GYM_SSH_ROOT")
        or "/root/daytona-training-gym-spec"
    )
    return target, ident, root


def sync_runs_from_ssh(
    *,
    target: str,
    local_runs: Path,
    remote_root: str | None = None,
    identity: Path | None = None,
    ssh_port: int | None = None,
) -> Path:
    """Mirror remote ``<root>/runs`` into ``local_runs``.

    Prefers PtyShell (works in Cursor agent shells without ``/dev/tty``).
    """
    local_runs = Path(local_runs).resolve()
    local_runs.mkdir(parents=True, exist_ok=True)

    host, embedded_port = parse_remote_target(target)
    port = resolve_ssh_port(ssh_port, embedded_port)
    ident = resolve_identity(identity)

    root = (
        remote_root
        or os.environ.get("DAYTONA_GYM_REMOTE_REPO")
        or os.environ.get("DAYTONA_GYM_SSH_ROOT")
        or "daytona-training-gym-spec"
    ).strip()
    root = root.removeprefix("~/").rstrip("/")
    abs_root = root if root.startswith("/") else f"/root/{root}"
    remote_runs = f"{abs_root}/runs"

    errors: list[str] = []

    try:
        sync_runs_via_pty(
            host=host,
            remote_root=abs_root,
            local_runs=local_runs,
            identity=ident,
            ssh_port=port,
        )
        return local_runs
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pty: {exc}")

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
            root=abs_root,
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
        "  • Prefer:  dg dash   (auto uses RUNPOD_* + SSH identity from .env)\n"
        "  • Escape:  dg dash --remote user@ssh.runpod.io -i ~/.ssh/<key>\n"
        "  • Or on the GPU box:  dg dash --export runs/dashboard.html".format(
            host=host, errs="\n".join(errors)
        )
    )


def sync_run_live_via_pty(
    *,
    host: str,
    remote_root: str,
    run_id: str,
    local_runs: Path,
    identity: Path | None = None,
    ssh_port: int | None = None,
    worker_log: str | None = None,
) -> dict[str, Any]:
    """Pull one run's progress.json + worker log tail (fast path for wait UI)."""
    from daytona_gym.gym.ssh_shell import PtyShell
    from daytona_gym.telemetry.progress import write_progress

    local_runs = Path(local_runs).resolve()
    local_runs.mkdir(parents=True, exist_ok=True)
    remote_root = remote_root.rstrip("/")
    progress_remote = f"{remote_root}/runs/{run_id}.progress.json"
    log_remote = (worker_log or "").strip()

    ssh_argv = [
        "ssh",
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=25",
    ]
    if identity is not None:
        ssh_argv.extend(["-i", str(identity)])
    if ssh_port is not None:
        ssh_argv.extend(["-p", str(ssh_port)])
    ssh_argv.append(host)

    # Base64 payloads; markers must be printed on their own lines (see
    # ``_slice_marked_block``) so the echoed command line is ignored.
    marker_begin = "__DG_LIVE_BEGIN__"
    marker_end = "__DG_LIVE_END__"
    jsonl_remote = f"{remote_root}/runs/{run_id}.jsonl"
    if log_remote:
        log_expr = (
            f"tail -n 80 {shlex.quote(log_remote)} 2>/dev/null "
            f"| base64 2>/dev/null | tr -d '\\n'"
        )
    else:
        log_expr = "printf ''"
    remote_cmd = (
        f"printf '%s\\n' {marker_begin}; "
        f"printf '%s\\n' __PROGRESS_B64__; "
        f"(base64 {shlex.quote(progress_remote)} 2>/dev/null | tr -d '\\n') || true; "
        f"printf '\\n%s\\n' __LOG_B64__; "
        f"{log_expr}; "
        f"printf '\\n%s\\n' __JSONL_B64__; "
        f"(base64 {shlex.quote(jsonl_remote)} 2>/dev/null | tr -d '\\n') || true; "
        f"printf '\\n%s\\n' {marker_end}"
    )
    with PtyShell(ssh_argv, connect_timeout=45) as shell:
        out = shell.run(remote_cmd, timeout=90)

    body = _slice_marked_block(out, marker_begin, marker_end)
    prog_b64 = _b64_section_after_label(body, "__PROGRESS_B64__", "__LOG_B64__")
    log_b64 = _b64_section_after_label(body, "__LOG_B64__", "__JSONL_B64__")
    jsonl_b64 = _b64_section_after_label(body, "__JSONL_B64__", None)

    prog: dict[str, Any] = {}
    if prog_b64:
        try:
            raw = base64.b64decode(prog_b64, validate=False).decode("utf-8", "replace")
            if "{" in raw:
                raw = raw[raw.index("{") :]
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                prog = loaded
        except Exception:  # noqa: BLE001
            prog = {}

    log_lines: list[str] = []
    if log_b64:
        try:
            log_text = base64.b64decode(log_b64, validate=False).decode(
                "utf-8", "replace"
            )
            log_lines = [
                ln.rstrip()
                for ln in log_text.splitlines()
                if ln.strip() and _is_useful_worker_log_line(ln)
            ][-40:]
        except Exception:  # noqa: BLE001
            log_lines = []

    if jsonl_b64:
        try:
            jsonl_bytes = base64.b64decode(jsonl_b64, validate=False)
            if jsonl_bytes.strip():
                (local_runs / f"{run_id}.jsonl").write_bytes(jsonl_bytes)
        except Exception:  # noqa: BLE001
            pass

    if prog:
        if log_lines:
            prog["log_tail"] = log_lines
        extra = {
            k: v
            for k, v in prog.items()
            if k
            not in {
                "run_id",
                "phase",
                "message",
                "updated_at",
            }
        }
        write_progress(
            local_runs,
            run_id,
            phase=str(prog.get("phase") or "boot"),
            message=str(prog.get("message") or "Worker running"),
            append_activity=False,
            **extra,
        )
    elif log_lines:
        last = log_lines[-1][:120]
        write_progress(
            local_runs,
            run_id,
            phase="boot",
            message=last,
            status="running",
            log_tail=log_lines,
            detail="From worker log (progress.json not readable yet)",
        )
    return prog


def _is_useful_worker_log_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    noise = (
        "tail -n",
        "base64",
        "echo __",
        "__DG_",
        "__PROGRESS",
        "__LOG",
        "2>/dev/null",
        "cat \"",
        "printf ",
    )
    if any(n in s for n in noise):
        return False
    if s.startswith("root@"):
        return False
    return True


def sync_runs_via_pty(
    *,
    host: str,
    remote_root: str,
    local_runs: Path,
    identity: Path | None = None,
    ssh_port: int | None = None,
) -> Path:
    """Pull ``remote_root/runs`` via ``PtyShell`` (no ``/dev/tty`` required)."""
    from daytona_gym.gym.ssh_shell import PtyShell

    local_runs = Path(local_runs).resolve()
    local_runs.mkdir(parents=True, exist_ok=True)
    remote_root = remote_root.rstrip("/")
    ssh_argv = [
        "ssh",
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=25",
    ]
    if identity is not None:
        ssh_argv.extend(["-i", str(identity)])
    if ssh_port is not None:
        ssh_argv.extend(["-p", str(ssh_port)])
    ssh_argv.append(host)

    marker_begin = "__DG_TAR_BEGIN__"
    marker_end = "__DG_TAR_END__"
    remote_cmd = (
        f"printf '%s\\n' {marker_begin}; "
        f'tar czf - -C "{remote_root}" runs 2>/dev/null | base64; '
        f"printf '\\n%s\\n' {marker_end}"
    )
    with PtyShell(ssh_argv, connect_timeout=60) as shell:
        out = shell.run(remote_cmd, timeout=180)

    payload = _slice_marked_block(out, marker_begin, marker_end)
    b64 = "".join(
        line.strip()
        for line in payload.splitlines()
        if line.strip()
        and not line.strip().startswith("--")
        and "echo " not in line
        and "printf " not in line
        and "base64" not in line
        and marker_begin not in line
        and marker_end not in line
    )
    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", b64)
    if not b64:
        return local_runs
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"base64 decode failed: {exc}") from exc
    if not raw:
        return local_runs
    with tempfile.NamedTemporaryFile(suffix=".tgz") as tmp:
        tmp.write(raw)
        tmp.flush()
        with tarfile.open(tmp.name, mode="r:gz") as archive:
            _safe_extract(archive, local_runs.parent)
    return local_runs


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
    """Legacy pull via ssh with a controlling TTY."""
    import shlex

    if root.startswith("/"):
        tar_cd = shlex.quote(root)
        remote_body = (
            "printf '%s\\n' __DG_TAR_BEGIN__; "
            f"tar czf - -C {tar_cd} runs | base64; "
            "printf '\\n%s\\n' __DG_TAR_END__"
        )
    else:
        remote_body = (
            "printf '%s\\n' __DG_TAR_BEGIN__; "
            f'tar czf - -C "$HOME/{root}" runs | base64; '
            "printf '\\n%s\\n' __DG_TAR_END__"
        )

    cmd = [
        "ssh",
        "-tt",
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

    try:
        tty_in = open("/dev/tty", "rb", buffering=0)  # noqa: SIM115
    except OSError as exc:
        raise RuntimeError(
            "no controlling terminal (/dev/tty). Use `dg dash` (PtyShell sync)."
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
    try:
        payload = _slice_marked_block(text, "__DG_TAR_BEGIN__", "__DG_TAR_END__")
    except RuntimeError as exc:
        err = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            err or f"ssh exit {rc}; {exc} (first 200 chars: {text[:200]!r})"
        ) from exc
    b64 = "".join(
        line.strip()
        for line in payload.splitlines()
        if line.strip()
        and not line.strip().startswith("--")
        and "printf " not in line
        and "base64" not in line
    )
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
