from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class CodingRecipe:
    """Coding-agent Slime×Daytona knobs proven in ``examples/coding_dogfood/``."""

    batch_size: int = 1
    n_samples: int = 1
    num_rollout: int = 1
    num_steps_per_rollout: int = 1
    max_concurrency: int | None = None
    max_response_len: int = 768
    temperature: float = 0.4
    seed_coding: bool = True
    seed_profile: str = "basic"
    bootstrap_run_tests: bool = True
    bootstrap_run_tests_cmd: str = "python test_broken.py"
    require_passing_tests: bool = True
    allow_aborted: bool = False
    timeout_seconds: float | None = None
    tool_timeout_seconds: float | None = None
    sandbox_timeout_seconds: float | None = None
    max_turns: int | None = None
    max_tools_per_turn: int | None = None
    generate_path: str = "daytona_gym.adapters.slime.generate_dogfood.generate"
    rm_path: str = "daytona_gym.adapters.slime.reward.reward"
    save_interval: int = 9999

    def effective_max_concurrency(self) -> int:
        if self.max_concurrency is not None:
            return int(self.max_concurrency)
        return int(self.batch_size) * int(self.n_samples)

    def global_batch_size(self) -> int:
        return int(self.batch_size) * int(self.n_samples)


def _recipe(**overrides: object) -> CodingRecipe:
    base = CodingRecipe()
    if not overrides:
        return base
    allowed = {f.name for f in fields(CodingRecipe)}
    unknown = set(overrides) - allowed
    if unknown:
        raise TypeError(f"unknown CodingRecipe fields: {sorted(unknown)}")
    return replace(base, **overrides)  # type: ignore[arg-type]


def Qwen25_3B_Recipe(**overrides: object) -> CodingRecipe:
    """Coding GRPO recipe paired with ``Qwen25_3B()`` (Modal-shaped name)."""
    return _recipe(**overrides)


def Qwen25_05B_Recipe(**overrides: object) -> CodingRecipe:
    """Coding GRPO recipe paired with ``Qwen25_05B()``."""
    return _recipe(**overrides)
