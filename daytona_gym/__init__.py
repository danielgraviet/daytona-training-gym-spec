"""Daytona Training Gym — portable rollout environments for existing RL trainers."""

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
    "DaytonaError",
    "EnvironmentHandle",
    "EnvironmentRuntime",
    "EnvironmentSpec",
    "ErrorCode",
    "FakeEnvironmentRuntime",
    "ToolAction",
    "ToolName",
    "ToolResult",
    "__version__",
]
