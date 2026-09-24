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
