from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _configure_ssl_certs() -> None:
    try:
        import certifi
    except ImportError:
        return
    cert_path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cert_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
    os.environ.setdefault("CURL_CA_BUNDLE", cert_path)


@pytest.fixture(scope="session", autouse=True)
def load_daytona_env() -> None:
    _configure_ssl_certs()
    _load_dotenv(_ENV_PATH)
    if not os.environ.get("DAYTONA_API_KEY"):
        pytest.skip("DAYTONA_API_KEY is not set (add it to .env or the environment)")
