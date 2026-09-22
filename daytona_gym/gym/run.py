from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingRun:
    """Handle returned by ``TrainConfig.build()`` / ``launch()``."""

    run_id: str
    telemetry_path: str
    command: list[str]
    env: dict[str, str]
    runtime_env: dict[str, Any]
    dry_run: bool = True
    returncode: int | None = None
    model_script: str = ""
    inspect_hint: str = ""

    def __post_init__(self) -> None:
        if not self.inspect_hint:
            self.inspect_hint = f"dg stats {self.telemetry_path}"
