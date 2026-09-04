from __future__ import annotations

from typing import Protocol


class Metrics(Protocol):
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


class NoOpMetrics:
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        return None

    def observe(self, name: str, value: float, **labels: str) -> None:
        return None
