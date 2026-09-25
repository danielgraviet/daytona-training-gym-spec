"""dg ingest deploy: re-deploys must actually change the running code."""

from __future__ import annotations

import subprocess

from daytona_gym.ingest import deploy

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_branch_is_pinned_to_commit(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr=""),
    )
    assert deploy.resolve_spec("main") == f"git+{deploy.REPO_URL}@{SHA}"
    assert deploy.resolve_spec(SHA) == f"git+{deploy.REPO_URL}@{SHA}"


def test_unresolvable_ref_falls_back_to_ref(monkeypatch) -> None:
    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert deploy.resolve_spec("v1.2") == f"git+{deploy.REPO_URL}@v1.2"


class _Resp:
    def __init__(self, exit_code: int = 0, result: str = "") -> None:
        self.exit_code = exit_code
        self.result = result


class _Proc:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, timeout: int | None = None) -> _Resp:
        self.commands.append(command)
        return _Resp()


class _Sandbox:
    def __init__(self) -> None:
        self.process = _Proc()


async def test_install_forces_reinstall_of_the_gym() -> None:
    sandbox = _Sandbox()
    await deploy.install_package(sandbox, f"git+{deploy.REPO_URL}@{SHA}")
    (cmd,) = sandbox.process.commands
    assert "--force-reinstall --no-deps" in cmd and SHA in cmd
