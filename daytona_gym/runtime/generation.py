from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from daytona_gym.runtime.errors import DaytonaError, ErrorCode


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    request_id: str | None = None


class GenerationBackend(Protocol):
    async def generate(self, conversation: str, sampling_params: dict) -> GenerationResult: ...


class ScriptedGenerator:
    """Deterministic generation backend for CPU contract tests."""

    def __init__(self, turns: Sequence[str | GenerationResult]) -> None:
        self._turns = list(turns)
        self._index = 0

    async def generate(self, conversation: str, sampling_params: dict) -> GenerationResult:
        del conversation, sampling_params
        if self._index >= len(self._turns):
            raise DaytonaError(
                ErrorCode.INFERENCE_FAILED,
                "scripted generator has no remaining turns",
            )
        item = self._turns[self._index]
        self._index += 1
        if isinstance(item, str):
            return GenerationResult(text=item)
        return item
