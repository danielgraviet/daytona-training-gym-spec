from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class CodingRecipe:
    """Slime×Daytona coding-agent knobs (dataset seeds drive the sandbox).

    Toy add(a,b) seed is **opt-in** via ``ToyCodingRecipe`` / ``seed_coding=True``.
    Prefer Harbor / JSONL rows with ``seed_files`` + ``run_tests_command``.
    """

    batch_size: int = 1
    n_samples: int = 1
    num_rollout: int = 1
    num_steps_per_rollout: int = 1
    max_concurrency: int | None = None
    max_response_len: int = 768
    temperature: float = 0.4
    # Opt-in toy seed (basic/multifile profiles). Off by default so dataset
    # rows without seed_files do not silently train on add(a,b).
    seed_coding: bool = False
    seed_profile: str = "basic"
    bootstrap_run_tests: bool = True
    # Fallback when the row has no run_tests_command. Empty by default —
    # ToyCodingRecipe sets ``python test_broken.py``.
    bootstrap_run_tests_cmd: str = ""
    require_passing_tests: bool = True
    allow_aborted: bool = False
    timeout_seconds: float | None = None
    tool_timeout_seconds: float | None = None
    sandbox_timeout_seconds: float | None = None
    max_turns: int | None = None
    max_tools_per_turn: int | None = None
    # Cap captured tool stdout/stderr (chars). Keeps observations short so
    # Megatron logits for the next train step stay on-GPU.
    tool_output_limit: int | None = None
    # Soft budget on prompt + generation + tool observation tokens for one
    # rollout. When exceeded the rollout ends as ``truncated`` instead of
    # growing until the trainer OOMs on a long sample.
    max_total_tokens: int | None = None
    # Daytona sandbox image / snapshot (snapshot preferred when set).
    image: str | None = None
    snapshot: str | None = None
    # Optional dotted path ``module.fn(args, sample) -> float`` (async OK).
    # Wired into generate as ``daytona_reward_function``; semantics stay user-owned.
    reward_path: str | None = None
    generate_path: str = "daytona_gym.adapters.slime.generate.generate"
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


def ToyCodingRecipe(**overrides: object) -> CodingRecipe:
    """Built-in add(a,b) sandbox seed — dogfood / smoke only."""
    return _recipe(
        seed_coding=True,
        seed_profile="basic",
        bootstrap_run_tests=True,
        bootstrap_run_tests_cmd="python test_broken.py",
        generate_path="daytona_gym.adapters.slime.generate_dogfood.generate",
        **overrides,
    )


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
