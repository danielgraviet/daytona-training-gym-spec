from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any

from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.traces import RecordingTracer


@dataclass
class FakeSlimeSample:
    """Duck-typed stand-in for slime.utils.types.Sample."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float | None = None
    loss_mask: list[int] | None = None
    status: Status = Status.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


def tool_turn(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"type": "tool", "name": name, "arguments": arguments})


def final_turn(content: str) -> str:
    return json.dumps({"type": "final", "content": content})


def make_args(
    *,
    runtime: EnvironmentRuntime | None = None,
    generator: ScriptedGenerator | None = None,
    tracer: RecordingTracer | None = None,
    **overrides: Any,
) -> SimpleNamespace:
    values = {
        "daytona_image": "fake-image:latest",
        "daytona_snapshot": None,
        "daytona_run_id": "run_test",
        "daytona_project_id": "coding-rl",
        "daytona_max_turns": 8,
        "daytona_timeout_seconds": None,
        "daytona_tool_timeout_seconds": None,
        "daytona_environment_runtime": runtime,
        "daytona_generator": generator,
        "daytona_tracer": tracer,
    }
    values.update(overrides)
    return SimpleNamespace(**values)
