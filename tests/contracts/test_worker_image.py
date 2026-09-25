"""3.1: the worker image resolves paths from its environment, not a git checkout."""

from __future__ import annotations

from pathlib import Path

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.models import Qwen25_3B, Qwen25_05B

REPO = Path(__file__).resolve().parents[2]


def test_model_paths_default_to_slime_layout(monkeypatch) -> None:
    for var in ("DAYTONA_GYM_MODELS_DIR", "HF_CHECKPOINT", "REF_LOAD"):
        monkeypatch.delenv(var, raising=False)
    m = Qwen25_3B()
    assert m.resolved_hf_checkpoint() == "/root/Qwen2.5-3B-Instruct/"
    assert m.resolved_ref_load() == "/root/Qwen2.5-3B-Instruct_torch_dist/"


def test_models_dir_is_resolved_on_the_worker(monkeypatch) -> None:
    monkeypatch.delenv("HF_CHECKPOINT", raising=False)
    monkeypatch.delenv("REF_LOAD", raising=False)
    m = Qwen25_05B()  # built on the laptop: no paths baked in
    monkeypatch.setenv("DAYTONA_GYM_MODELS_DIR", "/models/")
    assert m.resolved_hf_checkpoint() == "/models/Qwen2.5-0.5B-Instruct/"
    compute = m.to_compute()
    assert str(compute.ref_load) == "/models/Qwen2.5-0.5B-Instruct_torch_dist/"
    # Explicit paths still win.
    assert Qwen25_05B(hf_checkpoint="/data/hf/").resolved_hf_checkpoint() == "/data/hf/"


def test_workdir_env_beats_checkout(monkeypatch, tmp_path) -> None:
    c = LocalSlimeCompute(slime_root="/s", megatron_root="/m", hf_checkpoint="/h", ref_load="/r")
    monkeypatch.delenv("DAYTONA_GYM_WORKDIR", raising=False)
    assert c.resolved_repo() == REPO  # dev checkout
    monkeypatch.setenv("DAYTONA_GYM_WORKDIR", str(tmp_path))
    assert c.resolved_repo() == tmp_path.resolve()


def test_dockerignore_keeps_secrets_out_of_the_image() -> None:
    ignored = {line.strip() for line in (REPO / ".dockerignore").read_text().splitlines()}
    assert {".env", ".env.*", "scratch.txt", "runs/", ".git/"} <= ignored


def test_dockerfile_pins_base_by_digest() -> None:
    text = (REPO / "docker" / "worker" / "Dockerfile").read_text()
    assert "slimerl/slime@sha256:" in text
