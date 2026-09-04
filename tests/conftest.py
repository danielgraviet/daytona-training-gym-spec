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
