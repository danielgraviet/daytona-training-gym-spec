from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.launch import build_plan, execute_plan, plan_to_training_run
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.gym.run import TrainingRun
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


@dataclass
class TrainConfig:
    """Modal-shaped entry: model/compute + dataset + recipe → ``launch()``."""

    compute: LocalSlimeCompute
    dataset: PromptJsonlDataset
    recipe: CodingRecipe
    run_name: str | None = None
    telemetry_path: str | Path | None = None

    def validate(self, *, require_existing_paths: bool = True) -> None:
        if self.recipe.batch_size < 1 or self.recipe.n_samples < 1:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "batch_size and n_samples must be >= 1",
            )
        if require_existing_paths:
            missing: list[str] = []
            for label, path in (
                ("slime_root", self.compute.slime_root_path()),
                ("megatron_root", self.compute.megatron_root_path()),
                ("hf_checkpoint", self.compute.hf_checkpoint_path()),
                ("ref_load", self.compute.ref_load_path()),
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
    ) -> TrainingRun:
        """Start Slime on this BYO GPU host, or return the plan when ``dry_run``."""
        if dry_run:
            return self.build()
        self.validate(require_existing_paths=True)
        plan = build_plan(self)
        return execute_plan(
            plan,
            skip_preflight=skip_preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
        )
