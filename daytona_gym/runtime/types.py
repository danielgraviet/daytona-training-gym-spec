from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping, Protocol


class ToolName(StrEnum):
    RUN_COMMAND = "run_command"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPLY_PATCH = "apply_patch"
    RUN_TESTS = "run_tests"


@dataclass(frozen=True)
class EnvironmentSpec:
    image: str | None = None
    snapshot: str | None = None
    timeout_seconds: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentHandle:
    sandbox_id: str
    run_id: str
    rollout_id: str


@dataclass(frozen=True)
class ToolAction:
    name: ToolName
    arguments: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    truncated: bool = False


@dataclass(frozen=True)
class TrajectoryEvent:
    type: str
    text: str
    started_at: datetime
    finished_at: datetime
    tool_name: str | None = None
    exit_code: int | None = None
    ok: bool | None = None
    error_code: str | None = None


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class OrdinalTokenizer:
    """Deterministic CPU tokenizer used when no real tokenizer is configured."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]
