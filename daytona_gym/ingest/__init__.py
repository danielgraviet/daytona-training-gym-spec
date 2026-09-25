"""Durable, off-box run history.

The GPU worker runs a :class:`~daytona_gym.ingest.shipper.TelemetryShipper`
that tails ``runs/<id>.jsonl`` + ``runs/<id>.progress.json`` and pushes them to
an ingest service (``dg ingest``), which stores the same ``runs/`` layout and
serves the dashboard. Runs therefore outlive the pod, and a run whose worker
stops heartbeating shows ``worker_lost``.

Configure with ``DAYTONA_GYM_INGEST_URL`` + ``DAYTONA_GYM_INGEST_TOKEN``.
"""

from daytona_gym.ingest.config import IngestConfig
from daytona_gym.ingest.shipper import TelemetryShipper, start_shipper_from_env

__all__ = ["IngestConfig", "TelemetryShipper", "start_shipper_from_env"]
