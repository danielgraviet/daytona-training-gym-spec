"""Daytona Training Gym — portable rollout environments for existing RL trainers."""

from daytona_gym.gym import (
    CodingRecipe,
    LocalSlimeCompute,
    PromptJsonlDataset,
    Qwen25_05B,
    Qwen25_05B_Recipe,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    SoftSlimeModel,
    TrainConfig,
    TrainingRun,
)
from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.types import (
    EnvironmentHandle,
    EnvironmentSpec,
    ToolAction,
    ToolName,
    ToolResult,
)

__version__ = "0.1.0"

__all__ = [
    "CodingRecipe",
    "DaytonaEnvironmentRuntime",
    "DaytonaError",
    "EnvironmentHandle",
    "EnvironmentRuntime",
    "EnvironmentSpec",
    "ErrorCode",
    "FakeEnvironmentRuntime",
    "LocalSlimeCompute",
    "PromptJsonlDataset",
    "Qwen25_05B",
    "Qwen25_05B_Recipe",
    "Qwen25_3B",
    "Qwen25_3B_Recipe",
    "SoftSlimeModel",
    "ToolAction",
    "ToolName",
    "ToolResult",
    "TrainConfig",
    "TrainingRun",
    "__version__",
]
