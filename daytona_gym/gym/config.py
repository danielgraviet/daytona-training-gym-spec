from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.launch import build_plan, execute_plan, plan_to_training_run
from daytona_gym.gym.models import SoftSlimeModel
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.gym.run import TrainingRun
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

if TYPE_CHECKING:
    from daytona_gym.gym.worker import Worker


@dataclass
class TrainConfig:
    """Modal-shaped entry: ``model`` + ``dataset`` + ``recipe`` → ``launch()``.

    Prefer::

        TrainConfig(model=Qwen25_3B(), dataset=..., recipe=Qwen25_3B_Recipe())

    BYO remote GPU::

        TrainConfig(...).launch(worker=SshWorker(host="root@IP", port=..., identity="..."))

    Advanced: pass ``compute=LocalSlimeCompute(...)`` instead of ``model``.
    """

    dataset: PromptJsonlDataset
    recipe: CodingRecipe
    model: SoftSlimeModel | None = None
    compute: LocalSlimeCompute | None = None
    run_name: str | None = None
    telemetry_path: str | Path | None = None
    repo: str | Path | None = None

    def __post_init__(self) -> None:
        if self.model is None and self.compute is None:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "TrainConfig requires model=... (preferred) or compute=...",
            )

    def resolved_compute(self) -> LocalSlimeCompute:
        if self.compute is not None:
            return self.compute
        assert self.model is not None
        return self.model.to_compute(repo=self.repo)

    def validate(self, *, require_existing_paths: bool = True) -> None:
        if self.recipe.batch_size < 1 or self.recipe.n_samples < 1:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "batch_size and n_samples must be >= 1",
            )
        compute = self.resolved_compute()
        if require_existing_paths:
            missing: list[str] = []
            for label, path in (
                ("slime_root", compute.slime_root_path()),
                ("megatron_root", compute.megatron_root_path()),
                ("hf_checkpoint", compute.hf_checkpoint_path()),
                ("ref_load", compute.ref_load_path()),
                ("dataset", self.dataset.resolved_path()),
            ):
                if not path.exists():
                    missing.append(f"{label}={path}")
            if missing:
                raise DaytonaError(
                    ErrorCode.USER_CODE_ERROR,
                    "missing paths: " + ", ".join(missing),
                )

    def build(self) -> TrainingRun:
        """Build the Ray/Slime command without starting training (CPU-safe)."""
        plan = build_plan(self)
        return plan_to_training_run(plan, dry_run=True)

    def launch(
        self,
        *,
        worker: Worker | None = None,
        dry_run: bool = False,
        skip_preflight: bool = False,
        preflight_timeout_seconds: float = 90,
        open: bool | None = None,
        open_browser: bool = False,
        detach: bool = False,
    ) -> TrainingRun:
        """Start training on a BYO worker (local by default, or ``SshWorker``).

        ``open`` (default: True after a real launch) starts the live dashboard
        and sets ``run.dashboard_url`` (Cloudflare tunnel on RunPod/SSH).

        ``detach=True``: return as soon as ``run_id`` + ``dashboard_url`` are
        ready; training keeps running on the worker (Modal-shaped handle).
        Implies ``open=True`` (dashboard URL is the whole point).
        """
        from daytona_gym.gym.worker import LocalWorker

        if detach and open is False:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "detach=True requires a dashboard URL (do not pass open=False)",
            )
        if detach:
            open = True

        w: Worker = worker if worker is not None else LocalWorker()
        return w.launch(
            self,
            dry_run=dry_run,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
            open=open,
            open_browser=open_browser,
            detach=detach,
        )

    def _launch_local(
        self,
        *,
        dry_run: bool = False,
        skip_preflight: bool = False,
        preflight_timeout_seconds: float = 90,
        open: bool | None = None,
        open_browser: bool = False,
        detach: bool = False,
    ) -> TrainingRun:
        """On-box launch (used by ``LocalWorker`` and ``remote_job``)."""
        if dry_run:
            return self.build()
        if detach:
            # Supervised child holds dash + train so this process can return.
            import json
            import tempfile

            from daytona_gym.gym.remote_job import spawn_detached_run
            from daytona_gym.gym.worker import config_to_remote_payload

            repo = str(Path(self.repo).expanduser().resolve()) if self.repo else "."
            payload = config_to_remote_payload(
                self,
                remote_repo=repo,
                skip_preflight=skip_preflight,
                preflight_timeout_seconds=preflight_timeout_seconds,
                open=True if open is None else open,
                open_browser=open_browser,
            )
            payload["detach"] = True
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as fh:
                json.dump(payload, fh)
                job_path = Path(fh.name)
            return spawn_detached_run(job_path, payload)

        self.validate(require_existing_paths=True)
        plan = build_plan(self)
        run = execute_plan(
            plan,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
        )
        should_open = True if open is None else open
        if should_open:
            try:
                url = run.open(open_browser=open_browser)
                print(flush=True)
                print("=" * 60, flush=True)
                print("  ✓ training finished", flush=True)
                print(f"  ✓ run  {run.training_run_id}", flush=True)
                print(f"  → OPEN  {url}", flush=True)
                print("=" * 60, flush=True)
                print(flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"dashboard open failed: {exc}", file=sys.stderr)
                print(f"inspect: {run.inspect_hint}", file=sys.stderr)
        return run
