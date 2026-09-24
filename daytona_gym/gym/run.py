from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingRun:
    """Handle returned by ``TrainConfig.build()`` / ``launch()``.

    After a real launch::

        run = config.launch(open=True)   # prints live dashboard URL
        # or:
        run = config.launch()
        print(run.open())                # start dash + tunnel; returns URL
        run.wait_dashboard()             # block until Ctrl+C
    """

    run_id: str
    telemetry_path: str
    command: list[str]
    env: dict[str, str]
    runtime_env: dict[str, Any]
    dry_run: bool = True
    returncode: int | None = None
    model_script: str = ""
    inspect_hint: str = ""
    dashboard_url: str | None = None
    _dashboard: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.inspect_hint:
            self.inspect_hint = f"dg stats {self.telemetry_path}  |  dg dash"

    @property
    def training_run_id(self) -> str:
        """Modal-compatible alias for ``run_id``."""
        return self.run_id

    def open(
        self,
        *,
        share: bool | None = None,
        port: int = 8765,
        open_browser: bool = False,
    ) -> str:
        """Start the live dashboard (tunnel on RunPod/SSH by default).

        Returns a URL deep-linked to this run when possible. Keeps serving in
        the background until ``close_dashboard()`` / ``wait_dashboard()`` / process exit.
        """
        if self._dashboard is not None and self.dashboard_url:
            return self.dashboard_url

        from daytona_gym.telemetry.dashboard import start_dashboard

        runs_dir = Path(self.telemetry_path).expanduser().resolve().parent
        handle = start_dashboard(
            runs_dir=runs_dir,
            port=port,
            open_browser=open_browser,
            share=share,
            quiet=True,
        )
        stem = Path(self.telemetry_path).stem
        base = handle.url.rstrip("/")
        # Deep-link to this run's page when the jsonl exists under runs/
        deep = f"{base}/run/{stem}"
        self._dashboard = handle
        self.dashboard_url = deep
        self.inspect_hint = f"dg stats {self.telemetry_path}  |  {deep}"
        return deep

    def close_dashboard(self) -> None:
        """Stop the background dashboard / tunnel if running."""
        handle = self._dashboard
        self._dashboard = None
        if handle is not None:
            handle.stop()

    def wait_dashboard(self) -> None:
        """Block until Ctrl+C, then stop the dashboard."""
        if self._dashboard is None:
            return
        url = self.dashboard_url or self._dashboard.url
        print(flush=True)
        print("=" * 60, flush=True)
        print("  OPEN ON YOUR LAPTOP", flush=True)
        print(f"  {url}", flush=True)
        print("=" * 60, flush=True)
        print("  Ctrl+C to stop the dashboard", flush=True)
        print(flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nstopping dashboard", flush=True)
        finally:
            self.close_dashboard()
