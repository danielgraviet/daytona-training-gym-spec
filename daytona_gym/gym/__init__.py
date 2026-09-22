"""Modal-shaped gym facade: TrainConfig → local Slime×Daytona launch."""

from __future__ import annotations

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.config import TrainConfig
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.models import Qwen25_05B, Qwen25_3B, SoftSlimeModel
from daytona_gym.gym.recipe import (
    CodingRecipe,
    Qwen25_05B_Recipe,
    Qwen25_3B_Recipe,
)
from daytona_gym.gym.run import TrainingRun

__all__ = [
    "CodingRecipe",
    "LocalSlimeCompute",
    "PromptJsonlDataset",
    "Qwen25_05B",
    "Qwen25_05B_Recipe",
    "Qwen25_3B",
    "Qwen25_3B_Recipe",
    "SoftSlimeModel",
    "TrainConfig",
    "TrainingRun",
]
