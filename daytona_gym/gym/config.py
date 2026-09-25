from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.dataset import (
    AnyDataset,
    PromptJsonlDataset,
    materialize_dataset,
)
from daytona_gym.gym.harbor import HarborBackend, HarborRecipe, resolve_backend
from daytona_gym.gym.launch import (
    build_plan,
    execute_plan,
    plan_to_training_run,
    require_daytona_api_key,
)
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

        TrainConfig(...).launch(worker=runpod_worker(), detach=True)

    Advanced: pass ``compute=LocalSlimeCompute(...)`` instead of ``model``.

    ``backend=\"harbor\"`` selects the second gym adapter (not implemented yet).

    ``gpu_cost_per_hour`` (per GPU) turns idle-GPU time into dollars in the
    dashboard / ``dg stats``; omit it and $ figures are hidden.
    """

    dataset: AnyDataset
    recipe: CodingRecipe
    model: SoftSlimeModel | None = None
    compute: LocalSlimeCompute | None = None
    run_name: str | None = None
    telemetry_path: str | Path | None = None
    repo: str | Path | None = None
    backend: str = "slime"
    harbor: HarborBackend | HarborRecipe | None = None
    # Optional $/GPU-hour for the BYO worker; enables $ columns in analytics.
    gpu_cost_per_hour: float | None = None

    def __post_init__(self) -> None:
        self.backend = resolve_backend(self.backend)
        if self.backend == "harbor":
            return
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

    def ensure_dataset(self, *, runs_dir: Path | None = None) -> PromptJsonlDataset:
        """Materialize HF/Harbor datasets to JSONL; return a PromptJsonlDataset."""
        if isinstance(self.dataset, PromptJsonlDataset):
            return self.dataset
        compute = self.resolved_compute()
        repo = compute.resolved_repo()
        base = runs_dir or (repo / "runs")
        materialized = materialize_dataset(self.dataset, runs_dir=base)
        self.dataset = materialized
        return materialized

    def validate(self, *, require_existing_paths: bool = True) -> None:
        if self.recipe.batch_size < 1 or self.recipe.n_samples < 1:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "batch_size and n_samples must be >= 1",
            )
        compute = self.resolved_compute()
        if require_existing_paths:
            if not isinstance(self.dataset, PromptJsonlDataset):
                self.ensure_dataset()
            missing: list[str] = []
            for label, path in (
                ("slime_root", compute.slime_root_path()),
                ("megatron_root", compute.megatron_root_path()),
                ("hf_checkpoint", compute.hf_checkpoint_path()),
                ("ref_load", compute.ref_load_path()),
                ("dataset", self.dataset.resolved_path()),  # type: ignore[union-attr]
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

        ``detach=True``: return as soon as ``run_id`` is ready; training keeps
        running on the worker. Prefer laptop ``dashboard_url`` (localhost:3000).
        """
        from daytona_gym.gym.worker import LocalWorker

        if self.backend == "harbor":
            hb = self.harbor if isinstance(self.harbor, HarborBackend) else HarborBackend(
                recipe=self.harbor if isinstance(self.harbor, HarborRecipe) else HarborRecipe()
            )
            hb.launch(config=self, dry_run=dry_run)
            raise AssertionError("unreachable")

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

        import time

        from daytona_gym.telemetry.progress import write_progress

        # Progress file drives the dashboard's run status; write it for the
        # whole lifecycle (including a terminal state) like ``remote_job`` does.
        progress: dict[str, object] = {"runs_dir": None, "stem": None}
        started_at = time.time()

        def set_phase(phase: str, message: str, **extra: object) -> None:
            if progress["runs_dir"] is None:
                return
            try:
                write_progress(
                    progress["runs_dir"],  # type: ignore[arg-type]
                    str(progress["stem"]),
                    phase=phase,
                    message=message,
                    **extra,
                )
            except Exception:  # noqa: BLE001 — progress must never break training
                pass

        from daytona_gym.ingest.shipper import start_shipper_from_env

        run: TrainingRun | None = None
        dash_run: TrainingRun | None = None
        shipper = None
        try:
            # Plan first (paths only) so progress + shipping cover model prep too.
            plan = build_plan(self)
            progress["runs_dir"] = plan.telemetry_path.parent
            progress["stem"] = plan.telemetry_path.stem
            set_phase(
                "starting",
                "Launching Slime on this GPU host",
                status="running",
                started_at=started_at,
            )
            shipper = start_shipper_from_env(plan.telemetry_path, plan.run_id)
            require_daytona_api_key()

            if self.model is not None:
                from daytona_gym.gym.model_prep import ensure_model_ready

                ensure_model_ready(self.model)

            self.validate(require_existing_paths=True)

            # Open the dashboard BEFORE training so the run can be watched live.
            should_open = True if open is None else open
            if should_open:
                dash_run = plan_to_training_run(plan, dry_run=False)
                try:
                    dash_run.open(open_browser=open_browser)
                    from daytona_gym.gym.console_ui import banner_open

                    banner_open(str(dash_run.dashboard_url))
                except Exception as exc:  # noqa: BLE001
                    print(f"dashboard open failed: {exc}", file=sys.stderr)
                    dash_run = None

            run = execute_plan(
                plan,
                skip_preflight=skip_preflight,
                preflight_timeout_seconds=preflight_timeout_seconds,
                on_phase=lambda phase, message, **extra: set_phase(
                    phase, message, status="running", **extra
                ),
            )
        except BaseException as exc:
            code = getattr(exc, "code", None)
            label = f"[{code}] " if code is not None else ""
            set_phase(
                "failed",
                f"{label}{getattr(exc, 'message', None) or exc}"[:500],
                status="failed",
                returncode=2,
                done=True,
                elapsed_s=time.time() - started_at,
            )
            if dash_run is not None:
                dash_run.close_dashboard()
            if shipper is not None:
                shipper.stop()
            raise

        if dash_run is not None:
            run._dashboard = dash_run._dashboard
            run.dashboard_url = dash_run.dashboard_url
            run.inspect_hint = dash_run.inspect_hint
        rc = int(run.returncode or 0)
        run.status = "failed" if rc != 0 else "completed"
        set_phase(
            run.status,
            "Training complete" if rc == 0 else f"Training failed (exit={rc})",
            status=run.status,
            returncode=rc,
            done=True,
            elapsed_s=time.time() - started_at,
        )
        if shipper is not None:
            shipper.stop()  # drains the final status to the ingest host
        run.result()  # prints Modal-shaped completion banner (already done)
        return run
