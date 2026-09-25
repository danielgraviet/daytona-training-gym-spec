"""Tests for .env loading."""

from __future__ import annotations

import os
from pathlib import Path

from daytona_gym.envfile import load_dotenv


def test_load_dotenv_setdefault(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("ZZZ_TEST_KEY=from_file\nZZZ_EXISTING=file_val\n", encoding="utf-8")
    monkeypatch.setenv("ZZZ_EXISTING", "already")
    monkeypatch.delenv("ZZZ_TEST_KEY", raising=False)
    assert load_dotenv(env) == env
    assert os.environ["ZZZ_TEST_KEY"] == "from_file"
    assert os.environ["ZZZ_EXISTING"] == "already"  # not overridden


def test_dg_cli_loads_cwd_dotenv_without_overriding(tmp_path, monkeypatch) -> None:
    """Regression: `dg ingest deploy --daytona` said DAYTONA_API_KEY is required
    even though it was in .env."""
    import os

    from daytona_gym.envfile import load_default_dotenvs

    (tmp_path / ".env").write_text(
        "export DAYTONA_API_KEY=from_cwd\nDG_TEST_ALREADY_SET=from_file\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DG_TEST_ALREADY_SET", "from_shell")

    loaded = load_default_dotenvs()

    assert (tmp_path / ".env").resolve() in loaded
    assert os.environ["DAYTONA_API_KEY"] == "from_cwd"
    assert os.environ["DG_TEST_ALREADY_SET"] == "from_shell"
