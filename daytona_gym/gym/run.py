from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingRun:
    """Handle returned by ``TrainConfig.build()`` / ``launch()``.

    Modal-shaped usage::

        run = config.launch(worker=..., detach=True)
        print(run.training_run_id)
        print(run.dashboard_url)
        run.result()   # block until done → prints "Training complete: …"

    Local / attached launches already finish before ``launch()`` returns; call
    ``result()`` anyway to print the same completion banner.
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
    detached: bool = False
    status: str = "pending"  # pending | running | completed | failed
    _dashboard: Any = field(default=None, repr=False, compare=False)
    _started_at: float = field(default_factory=time.time, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.inspect_hint:
            self.inspect_hint = f"dg stats {self.telemetry_path}  |  dg dash"
        if self.returncode is not None and self.status == "pending":
            self.status = "failed" if int(self.returncode) != 0 else "completed"
        elif self.detached and self.status == "pending":
            self.status = "running"

    @property
    def training_run_id(self) -> str:
        """Modal-compatible alias for ``run_id``."""
        return self.run_id

    def open(
        self,
        *,
        share: bool | None = None,
        port: int = 3000,
        open_browser: bool = False,
    ) -> str:
        """Start the live dashboard on loopback (Cloudflare only if ``share=True``).

        Returns a URL deep-linked to this run when possible. Keeps serving in
        the background until ``close_dashboard()`` / ``wait_dashboard()`` / process exit.
        Prefer laptop ``dg dash`` (auto RunPod sync) over worker tunnels.
        """
        if self._dashboard is not None and self.dashboard_url:
            return self.dashboard_url

        from daytona_gym.telemetry.dashboard import start_dashboard

        runs_dir = Path(self.telemetry_path).expanduser().resolve().parent
        want_share = bool(share)
        handle = start_dashboard(
            runs_dir=runs_dir,
            port=port,
            open_browser=open_browser,
            share=want_share,
            quiet=True,
        )
        stem = Path(self.telemetry_path).stem
        if want_share:
            base = str(handle.url).rstrip("/")
        else:
            base = str(getattr(handle, "local_url", None) or handle.url).rstrip("/")
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
        print("  dashboard (Ctrl+C to stop)", flush=True)
        print(f"  {url}", flush=True)
        print("=" * 60, flush=True)
        print(flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nstopping dashboard", flush=True)
        finally:
            self.close_dashboard()

    def done(self) -> bool:
        """True if training has finished (completed or failed). Does not stop the worker."""
        if self.status in {"completed", "failed"}:
            return True
        if self.returncode is not None and not self.detached:
            return True
        snap = self._poll_snapshot()
        if snap is None:
            return False
        self._apply_snapshot(snap)
        return self.status in {"completed", "failed"}

    def wait(self, *, timeout: float | None = None, poll_interval: float = 2.0) -> TrainingRun:
        """Block until training finishes. Does not tear down a detached worker.

        Polls the live dashboard API when ``dashboard_url`` is set, else local
        progress / status files. Shows a Rich spinner with phase heartbeats.
        """
        from daytona_gym.gym.console_ui import WaitProgress

        if self.dry_run:
            self.status = "completed"
            self.returncode = 0
            return self
        if self.done():
            return self

        # Dashboard URL already printed at launch when laptop dash started.
        # Do not print a second "OPEN" banner here.

        deadline = None if timeout is None else time.monotonic() + timeout
        ui = WaitProgress(self.training_run_id, started_at=self._started_at)
        ui.update(
            {
                "phase": "waiting",
                "message": "Waiting for worker phases (dataset → model → Ray/Slime)",
            }
        )
        try:
            while True:
                snap = self._poll_snapshot()
                ui.update(snap)
                if snap is not None:
                    self._apply_snapshot(snap)
                    if self.status in {"completed", "failed"}:
                        ui.finish()
                        return self
                if deadline is not None and time.monotonic() >= deadline:
                    ui.finish()
                    raise TimeoutError(
                        f"Timed out after {timeout}s waiting for "
                        f"training_run_id={self.training_run_id}"
                    )
                time.sleep(max(0.5, float(poll_interval)))
        except KeyboardInterrupt:
            ui.finish()
            print(flush=True)
            print(
                f'Disconnected from training "{self.training_run_id}". '
                "The worker was left running.",
                flush=True,
            )
            if self.dashboard_url:
                print(self.dashboard_url, flush=True)
            raise

    def result(self, *, timeout: float | None = None) -> TrainingRun:
        """Wait until done, then print a Modal-shaped completion banner."""
        from daytona_gym.gym.console_ui import banner_complete

        self.wait(timeout=timeout)
        ok = self.status == "completed" and int(self.returncode or 0) == 0
        banner_complete(
            ok=ok,
            run_id=self.training_run_id,
            returncode=self.returncode,
            url=self.dashboard_url,
        )
        return self

    def _print_completion_banner(self) -> None:
        from daytona_gym.gym.console_ui import banner_complete

        ok = self.status == "completed" and int(self.returncode or 0) == 0
        banner_complete(
            ok=ok,
            run_id=self.training_run_id,
            returncode=self.returncode,
            url=self.dashboard_url,
        )

    def _live_api_url(self) -> str | None:
        if not self.dashboard_url:
            return None
        raw = self.dashboard_url.rstrip("/")
        # deep link: https://host/run/<stem>  →  https://host/api/runs/<stem>/live
        if "/run/" in raw:
            base, _, stem = raw.partition("/run/")
            stem = stem.split("/", 1)[0]
            if base and stem:
                return f"{base}/api/runs/{stem}/live"
        stem = Path(self.telemetry_path).stem
        return f"{raw}/api/runs/{stem}/live"

    def _poll_snapshot(self) -> dict[str, Any] | None:
        url = self._live_api_url()
        if url:
            snap = self._http_json(url)
            if snap is not None:
                return snap
        # Local / on-worker progress file
        try:
            from daytona_gym.telemetry.progress import read_progress

            path = Path(self.telemetry_path).expanduser()
            prog = read_progress(path.parent, path.stem)
            if isinstance(prog, dict):
                return {
                    "phase": prog.get("phase"),
                    "message": prog.get("message"),
                    "detail": prog.get("detail"),
                    "status": prog.get("status"),
                    "done": prog.get("status") in {"completed", "failed"}
                    or prog.get("phase") in {"completed", "failed"},
                    "failed": prog.get("phase") == "failed"
                    or prog.get("status") == "failed",
                    "returncode": prog.get("returncode"),
                    "updated_at": prog.get("updated_at"),
                }
        except Exception:  # noqa: BLE001
            pass
        # Detached supervised status file (local detach)
        status_path = self.runtime_env.get("status_path")
        if status_path:
            try:
                data = json.loads(Path(str(status_path)).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict) and data.get("done"):
                return {
                    "phase": "failed" if data.get("error") else "completed",
                    "message": data.get("error") or "Training complete",
                    "status": "failed" if data.get("error") else "completed",
                    "done": True,
                    "failed": bool(data.get("error")),
                    "returncode": data.get("returncode", 1 if data.get("error") else 0),
                }
        return None

    @staticmethod
    def _http_json(url: str) -> dict[str, Any] | None:
        try:
            req = urllib.request.Request(
                url + (("&" if "?" in url else "?") + f"t={int(time.time())}"),
                headers={"User-Agent": "daytona-gym/0.1", "Cache-Control": "no-store"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _apply_snapshot(self, snap: dict[str, Any]) -> None:
        status = snap.get("status")
        phase = str(snap.get("phase") or "")
        if snap.get("failed") or status == "failed" or phase == "failed":
            self.status = "failed"
            if snap.get("returncode") is not None:
                self.returncode = int(snap["returncode"])
            elif self.returncode is None:
                self.returncode = 1
        elif snap.get("done") or status == "completed" or phase == "completed":
            self.status = "completed"
            if snap.get("returncode") is not None:
                self.returncode = int(snap["returncode"])
            elif self.returncode is None:
                self.returncode = 0
        elif phase or status:
            self.status = "running"
