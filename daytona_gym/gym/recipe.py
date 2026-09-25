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
    # Colocated Slime offloads the idle engine to CPU RAM each step by default
    # (None = Slime's default). False keeps both resident on the GPU: no host
    # RAM needed for offload and no switching cost, but both must fit in VRAM.
    # Used for 16 GB-RAM home boxes (RTX 3090 + 0.5B), see box_worker.py.
    offload_train: bool | None = None
    offload_rollout: bool | None = None

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


def Qwen25_7B_Recipe(**overrides: object) -> CodingRecipe:
    """Coding GRPO recipe paired with ``Qwen25_7B()``."""
    return _recipe(**{"max_response_len": 1024, **overrides})


def Qwen25_14B_Recipe(**overrides: object) -> CodingRecipe:
    """Coding GRPO recipe paired with ``Qwen25_14B()``."""
    return _recipe(**{"max_response_len": 1536, **overrides})
