"""Modal-shaped gym facade: TrainConfig → local Slime×Daytona launch."""

from __future__ import annotations

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.config import TrainConfig
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.gym.run import TrainingRun

__all__ = [
    "CodingRecipe",
    "LocalSlimeCompute",
    "PromptJsonlDataset",
    "TrainConfig",
    "TrainingRun",
]
