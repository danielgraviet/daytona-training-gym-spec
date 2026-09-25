"""Load repo ``.env`` into ``os.environ`` (does not override existing vars)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str | None = None) -> Path | None:
    """Read ``KEY=VALUE`` lines into the environment.

    Returns the path loaded, or ``None`` if missing.
    """
    if path is None:
        # daytona_gym/envfile.py → repo root
        path = Path(__file__).resolve().parents[1] / ".env"
    else:
        path = Path(path)
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)
    return path


_REPO_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_default_dotenvs() -> list[Path]:
    """Load ``./.env`` (where the user runs ``dg``) then the repo-root ``.env``.

    Existing environment variables always win; the first file wins over the
    second for keys set in both. Set ``DAYTONA_GYM_NO_DOTENV=1`` to skip.
    """
    if os.environ.get("DAYTONA_GYM_NO_DOTENV") == "1":
        return []
    loaded: list[Path] = []
    seen: set[Path] = set()
    for candidate in (Path.cwd() / ".env", _REPO_ENV):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        path = load_dotenv(resolved)
        if path is not None:
            loaded.append(path)
    return loaded
