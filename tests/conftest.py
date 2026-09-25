from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run live Daytona API tests (also collected when tests/e2e is passed explicitly)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: live Daytona API tests")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if "e2e" not in Path(collection_path).parts:
        return None
    if config.getoption("--e2e"):
        return False
    if any("e2e" in str(arg) for arg in config.args):
        return False
    return True


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's .env / shell ingest config out of unit tests.

    `dg` auto-loads ./.env and the repo .env; tests run from the repo root, so
    without this a real DAYTONA_GYM_INGEST_URL/TOKEN changes launch behavior.
    """
    monkeypatch.setenv("DAYTONA_GYM_NO_DOTENV", "1")
    monkeypatch.delenv("DAYTONA_GYM_INGEST_URL", raising=False)
    monkeypatch.delenv("DAYTONA_GYM_INGEST_TOKEN", raising=False)
