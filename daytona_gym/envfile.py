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
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)
    return path
