"""Harbor as a second gym backend (adapter boundary; not CLI-first).

Partners use ``TrainConfig(..., backend=\"harbor\")`` once wired. Harbor /
Terminal-Bench types stay out of core runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from daytona_gym.runtime.errors import DaytonaError, ErrorCode

GymBackendName = Literal["slime", "harbor"]


class HarborJob(Protocol):
    """Minimal Harbor-shaped job handle (framework-owned outer loop)."""

    job_id: str

    def wait(self) -> int: ...


@dataclass(frozen=True)
class HarborRecipe:
    """Placeholder recipe knobs for a future Harbor×Daytona path.

    Not a Terminal-Bench task definition — only gym-facing correlation /
    telemetry preferences.
    """

    task_name: str = "harbor"
    daytona_env: str = "daytona"
    emit_spans: bool = True
    max_trials: int = 1


@dataclass
class HarborBackend:
    """Second adapter under the gym facade (spike / contract surface).

    Full Harbor integration is deferred; this raises a typed error until the
    Harbor plugin path is wired. See ``HARBOR_INTEGRATION.md``.
    """

    recipe: HarborRecipe | None = None
    extra: dict[str, Any] | None = None

    def launch(self, **_: Any) -> HarborJob:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "Harbor backend is not implemented yet — use backend='slime' "
            "(TrainConfig + Slime). See HARBOR_INTEGRATION.md.",
        )


def resolve_backend(name: GymBackendName | str = "slime") -> str:
    key = str(name or "slime").strip().lower()
    if key not in {"slime", "harbor"}:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"unknown gym backend: {name!r} (expected 'slime' or 'harbor')",
        )
    return key
