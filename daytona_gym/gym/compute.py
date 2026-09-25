from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalSlimeCompute:
    """BYO GPU host paths for a local Slime + Megatron layout (RunPod / homelab)."""

    slime_root: str | Path
    megatron_root: str | Path
    hf_checkpoint: str | Path
    ref_load: str | Path
    model_script: str = "qwen2.5-0.5B.sh"
    repo: str | Path | None = None
    sglang_mem_fraction: float = 0.4
    save_dir: str | Path | None = None
    api_url: str = "https://app.daytona.io/api"
    num_gpus: int = 1

    def resolved_repo(self) -> Path:
        """Working directory for ``runs/`` (and the repo when running from a checkout).

        Order: explicit ``repo`` → ``$DAYTONA_GYM_WORKDIR`` → the git checkout the
        package lives in → the current directory (installed package, e.g. the
        worker image, where site-packages is not a place to write runs).
        """
        if self.repo is not None:
            return Path(self.repo).expanduser().resolve()
        workdir = os.environ.get("DAYTONA_GYM_WORKDIR")
        if workdir:
            return Path(workdir).expanduser().resolve()
        checkout = Path(__file__).resolve().parents[2]
        if (checkout / "pyproject.toml").is_file() and (checkout / "daytona_gym").is_dir():
            return checkout
        return Path.cwd().resolve()

    def slime_root_path(self) -> Path:
        return Path(self.slime_root).expanduser().resolve()

    def megatron_root_path(self) -> Path:
        return Path(self.megatron_root).expanduser().resolve()

    def hf_checkpoint_path(self) -> Path:
        return Path(self.hf_checkpoint).expanduser().resolve()

    def ref_load_path(self) -> Path:
        return Path(self.ref_load).expanduser().resolve()

    def save_dir_path(self) -> Path:
        if self.save_dir is not None:
            return Path(self.save_dir).expanduser().resolve()
        return Path("/tmp/slime_coding_dogfood_save")

    def model_script_path(self) -> Path:
        return self.slime_root_path() / "scripts" / "models" / self.model_script
