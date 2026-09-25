"""Secrets forwarded to BYO workers must never appear in payloads, command
lines, or captured shell output."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

from daytona_gym import SshWorker
from daytona_gym.gym.worker import (
    _export_forward_env_cmd,
    _install_forward_env_via_shell,
    config_to_remote_payload,
)
from tests.contracts.test_worker import _config

SECRET = "dtn_supersecret_value_123"


def test_payload_carries_key_names_not_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_API_KEY", SECRET)
    payload = config_to_remote_payload(
        _config(tmp_path),
        remote_repo="/root/repo",
        skip_preflight=True,
        preflight_timeout_seconds=30,
        open=False,
        open_browser=False,
    )
    blob = json.dumps(payload)
    assert SECRET not in blob
    assert "forward_env" not in payload
    assert "DAYTONA_API_KEY" in payload["forward_env_keys"]


def test_export_cmd_sources_then_deletes_env_file(monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_API_KEY", SECRET)
    cmd = _export_forward_env_cmd()
    assert SECRET not in cmd
    assert ". /tmp/daytona_gym_forward.env" in cmd
    assert "rm -f /tmp/daytona_gym_forward.env" in cmd


class _RecordingShell:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self._on_output = self._fail_if_called
        self.echo = True
        self.echoed_while_writing: list[str] = []

    def _fail_if_called(self, text: str) -> None:  # pragma: no cover - guard
        raise AssertionError(f"shell output leaked to local stdout: {text!r}")

    def run(self, command: str, **_: object) -> str:
        self.commands.append(command)
        if command == "stty -echo":
            self.echo = False
        elif command == "stty echo":
            self.echo = True
        elif self.echo:
            self.echoed_while_writing.append(command)
        return ""


def test_pty_install_hides_secret_and_disables_echo(monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_API_KEY", SECRET)
    shell = _RecordingShell()
    _install_forward_env_via_shell(shell)

    joined = "\n".join(shell.commands)
    assert SECRET not in joined
    assert shell.commands[0] == "stty -echo"
    assert shell.commands[-1] == "stty echo"
    # Nothing (including reversible base64 chunks) typed while echo was on.
    assert shell.echoed_while_writing == []
    assert any("umask 077" in c for c in shell.commands)
    # Muting is restored afterwards.
    assert shell._on_output == shell._fail_if_called

    # Sanity: the chunks do decode back to the env file body.
    chunks = [
        c.split("'")[3] for c in shell.commands if c.startswith("printf '%s' '")
    ]
    body = base64.b64decode("".join(chunks)).decode()
    assert f"DAYTONA_API_KEY={SECRET}" in body


def test_exec_path_never_puts_secret_on_argv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_API_KEY", SECRET)
    argvs: list[list[str]] = []
    env_file_modes: list[int] = []

    def fake_run(cmd, capture_output=False, text=False):  # noqa: ANN001
        argvs.append(list(cmd))
        for arg in cmd:
            if str(arg).endswith("forward.env") and Path(arg).exists():
                env_file_modes.append(Path(arg).stat().st_mode & 0o777)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    class FakePopen:
        def __init__(self, cmd, **_: object) -> None:
            argvs.append(list(cmd))
            self.stdout = iter(["__DG_RUN_ID__=run_x\n", "__DG_RETURNCODE__=0\n"])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr("daytona_gym.gym.worker._sync_package_via_scp", lambda _w: None)

    worker = SshWorker(host="root@1.2.3.4", port=22, identity="/tmp/key", pull=False, transport="exec")
    _config(tmp_path).launch(worker=worker, dry_run=False, open=False)

    assert argvs
    for argv in argvs:
        assert all(SECRET not in str(a) for a in argv)
    assert env_file_modes == [0o600]
