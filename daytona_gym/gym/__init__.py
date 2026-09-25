"""Modal-shaped gym facade: TrainConfig → local Slime×Daytona launch."""

from __future__ import annotations

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.config import TrainConfig
from daytona_gym.gym.dataset import HarborDataset, HuggingFaceDataset, PromptJsonlDataset
from daytona_gym.gym.harbor import HarborBackend, HarborRecipe
from daytona_gym.gym.models import (
    Qwen25_05B,
    Qwen25_3B,
    Qwen25_7B,
    Qwen25_14B,
    SoftSlimeModel,
)
from daytona_gym.gym.recipe import (
    CodingRecipe,
    Qwen25_05B_Recipe,
    Qwen25_3B_Recipe,
    Qwen25_7B_Recipe,
    Qwen25_14B_Recipe,
)
from daytona_gym.gym.run import TrainingRun
from daytona_gym.gym.worker import LocalWorker, SshWorker

__all__ = [
    "CodingRecipe",
    "HarborBackend",
    "HarborDataset",
    "HarborRecipe",
    "HuggingFaceDataset",
    "LocalSlimeCompute",
    "LocalWorker",
    "PromptJsonlDataset",
    "Qwen25_05B",
    "Qwen25_05B_Recipe",
    "Qwen25_3B",
    "Qwen25_3B_Recipe",
    "Qwen25_7B",
    "Qwen25_7B_Recipe",
    "Qwen25_14B",
    "Qwen25_14B_Recipe",
    "SoftSlimeModel",
    "SshWorker",
    "TrainConfig",
    "TrainingRun",
]
