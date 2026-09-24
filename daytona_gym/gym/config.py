from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.launch import build_plan, execute_plan, plan_to_training_run
from daytona_gym.gym.models import SoftSlimeModel
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.gym.run import TrainingRun
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


@dataclass
class TrainConfig:
    """Modal-shaped entry: ``model`` + ``dataset`` + ``recipe`` → ``launch()``.

    Prefer::

        TrainConfig(model=Qwen25_3B(), dataset=..., recipe=Qwen25_3B_Recipe())

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
        dry_run: bool = False,
        skip_preflight: bool = False,
        preflight_timeout_seconds: float = 90,
        open: bool | None = None,
        open_browser: bool = False,
    ) -> TrainingRun:
        """Start Slime on this BYO GPU host, or return the plan when ``dry_run``.

        ``open`` (default: True after a real launch) starts the live dashboard
        and sets ``run.dashboard_url`` (Cloudflare tunnel on RunPod/SSH).
        """
        if dry_run:
            return self.build()
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
                # Loud banner — easy to spot after Ray's verbose job logs.
                print(flush=True)
                print("=" * 60, flush=True)
                print("  ✓ training finished", flush=True)
                print(f"  ✓ run  {run.training_run_id}", flush=True)
                print(f"  → OPEN  {url}", flush=True)
                print("=" * 60, flush=True)
                print(flush=True)
            except Exception as exc:  # noqa: BLE001 — launch succeeded; dash is best-effort
                print(f"dashboard open failed: {exc}", file=sys.stderr)
                print(f"inspect: {run.inspect_hint}", file=sys.stderr)
        return run
