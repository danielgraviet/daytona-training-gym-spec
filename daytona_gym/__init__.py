"""Daytona Training Gym — portable rollout environments for existing RL trainers."""

from daytona_gym.gym import (
    CodingRecipe,
    HarborBackend,
    HarborDataset,
    HarborRecipe,
    HuggingFaceDataset,
    LocalSlimeCompute,
    LocalWorker,
    PromptJsonlDataset,
    Qwen25_05B,
    Qwen25_05B_Recipe,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    Qwen25_7B,
    Qwen25_7B_Recipe,
    Qwen25_14B,
    Qwen25_14B_Recipe,
    SoftSlimeModel,
    SshWorker,
    TrainConfig,
    TrainingRun,
)
from daytona_gym.gym.providers.runpod import create_pod, runpod_worker
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
    "ToolAction",
    "ToolName",
    "ToolResult",
    "TrainConfig",
    "TrainingRun",
    "create_pod",
    "runpod_worker",
    "__version__",
]
