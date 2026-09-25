"""Model presets for slime:latest BYO layouts (Modal-shaped public API)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from daytona_gym.gym.compute import LocalSlimeCompute


def models_dir() -> str:
    """Where HF checkpoints + converted Megatron weights live on the worker.

    Mount a persistent volume here in containers (``-v ~/gym-models:/models
    -e DAYTONA_GYM_MODELS_DIR=/models``) so weights survive restarts.
    """
    return (os.environ.get("DAYTONA_GYM_MODELS_DIR") or "/root").rstrip("/")


@dataclass(frozen=True)
class SoftSlimeModel:
    """HF + Slime model-script preset with default paths for ``slimerl/slime:latest``."""

    name: str
    hf_repo: str
    model_script: str
    # Empty = resolve on the machine that trains: $DAYTONA_GYM_MODELS_DIR/<name>/
    # (default /root, the slimerl/slime layout). Explicit paths still win.
    hf_checkpoint: str = ""
    ref_load: str = ""
    slime_root: str = "/root/slime"
    megatron_root: str = "/root/Megatron-LM"
    sglang_mem_fraction: float = 0.4
    save_dir: str = "/tmp/slime_coding_dogfood_save"
    api_url: str = "https://app.daytona.io/api"
    num_gpus: int = 1

    def resolved_hf_checkpoint(self) -> str:
        return os.environ.get("HF_CHECKPOINT") or self.hf_checkpoint or f"{models_dir()}/{self.name}/"

    def resolved_ref_load(self) -> str:
        return (
            os.environ.get("REF_LOAD")
            or self.ref_load
            or f"{models_dir()}/{self.name}_torch_dist/"
        )

    def to_compute(self, *, repo: str | Path | None = None) -> LocalSlimeCompute:
        """Materialize ``LocalSlimeCompute`` (escape hatch under the Modal-shaped API)."""
        return LocalSlimeCompute(
            slime_root=os.environ.get("SLIME_ROOT", self.slime_root),
            megatron_root=os.environ.get("MEGATRON_ROOT", self.megatron_root),
            hf_checkpoint=self.resolved_hf_checkpoint(),
            ref_load=self.resolved_ref_load(),
            model_script=os.environ.get("MODEL_SCRIPT", self.model_script),
            repo=repo,
            sglang_mem_fraction=float(
                os.environ.get("SGLANG_MEM", str(self.sglang_mem_fraction))
            ),
            save_dir=os.environ.get("SLIME_SAVE_DIR", self.save_dir),
            api_url=os.environ.get("DAYTONA_API_URL", self.api_url),
            num_gpus=int(os.environ.get("NUM_GPUS", str(self.num_gpus))),
        )


def Qwen25_3B(**overrides: object) -> SoftSlimeModel:
    """Qwen2.5-3B-Instruct — proven coding tool-loop on H100 dogfood."""
    base = SoftSlimeModel(
        name="Qwen2.5-3B-Instruct",
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        model_script="qwen2.5-3B.sh",
    )
    return replace(base, **overrides) if overrides else base  # type: ignore[arg-type]


def Qwen25_05B(**overrides: object) -> SoftSlimeModel:
    """Qwen2.5-0.5B-Instruct — smaller smoke on tight VRAM."""
    base = SoftSlimeModel(
        name="Qwen2.5-0.5B-Instruct",
        hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
        model_script="qwen2.5-0.5B.sh",
    )
    return replace(base, **overrides) if overrides else base  # type: ignore[arg-type]


def Qwen25_7B(**overrides: object) -> SoftSlimeModel:
    """Qwen2.5-7B-Instruct — larger coding tool-loop (needs more VRAM)."""
    base = SoftSlimeModel(
        name="Qwen2.5-7B-Instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        model_script="qwen2.5-7B.sh",
        sglang_mem_fraction=0.35,
    )
    return replace(base, **overrides) if overrides else base  # type: ignore[arg-type]


def Qwen25_14B(**overrides: object) -> SoftSlimeModel:
    """Qwen2.5-14B-Instruct — multi-GPU coding preset (set ``num_gpus``)."""
    base = SoftSlimeModel(
        name="Qwen2.5-14B-Instruct",
        hf_repo="Qwen/Qwen2.5-14B-Instruct",
        model_script="qwen2.5-14B.sh",
        sglang_mem_fraction=0.3,
        num_gpus=2,
    )
    return replace(base, **overrides) if overrides else base  # type: ignore[arg-type]


# Back-compat alias.
SlimeModelPreset = SoftSlimeModel
