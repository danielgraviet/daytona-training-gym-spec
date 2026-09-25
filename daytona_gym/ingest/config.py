"""Ingest endpoint configuration (env-driven so it forwards to BYO workers)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

INGEST_URL_ENV = "DAYTONA_GYM_INGEST_URL"
INGEST_TOKEN_ENV = "DAYTONA_GYM_INGEST_TOKEN"

# A run whose worker has not been heard from this long (and has no terminal
# status) is reported as ``worker_lost``. Shipper heartbeats every 15s.
WORKER_LOST_AFTER_SECONDS = 90.0

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def valid_run_id(run_id: str) -> bool:
    """Run ids become file names on the ingest host — no traversal, no oddities."""
    return bool(_RUN_ID_RE.match(run_id)) and ".." not in run_id


@dataclass(frozen=True)
class IngestConfig:
    url: str
    token: str

    @classmethod
    def from_env(cls) -> IngestConfig | None:
        url = (os.environ.get(INGEST_URL_ENV) or "").strip().rstrip("/")
        token = (os.environ.get(INGEST_TOKEN_ENV) or "").strip()
        if not url or not token:
            return None
        return cls(url=url, token=token)

    @classmethod
    def require_consistent_env(cls) -> IngestConfig | None:
        """Like :meth:`from_env`, but half a config is an error, not "off".

        Regression: a pod with only one of the two variables silently fell back
        to a local tunnel and the run never reached the ingest host.
        """
        url = bool((os.environ.get(INGEST_URL_ENV) or "").strip())
        token = bool((os.environ.get(INGEST_TOKEN_ENV) or "").strip())
        if url != token:
            from daytona_gym.runtime.errors import DaytonaError, ErrorCode

            missing = INGEST_TOKEN_ENV if url else INGEST_URL_ENV
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                f"{missing} is not set but its pair is — set both "
                f"{INGEST_URL_ENV} and {INGEST_TOKEN_ENV} (or neither) and relaunch.",
            )
        return cls.from_env()

    def run_url(self, run_id: str) -> str:
        return f"{self.url}/run/{run_id}"

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}
