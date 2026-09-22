"""CPU contract tests for the Gym SDK build / dry_run path."""

from __future__ import annotations

from pathlib import Path

import pytest

from daytona_gym import (
    CodingRecipe,
    LocalSlimeCompute,
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
)
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def _config_modal(tmp_path: Path) -> TrainConfig:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"fix","label":"1"}\n', encoding="utf-8")
    return TrainConfig(
        model=Qwen25_3B(
            slime_root=str(tmp_path / "slime"),
            megatron_root=str(tmp_path / "Megatron-LM"),
            hf_checkpoint=str(tmp_path / "hf"),
            ref_load=str(tmp_path / "ref"),
        ),
        dataset=PromptJsonlDataset(prompts),
        recipe=Qwen25_3B_Recipe(
            batch_size=2,
            n_samples=2,
            num_rollout=3,
            max_concurrency=4,
            allow_aborted=True,
            max_tools_per_turn=3,
        ),
        repo=tmp_path / "repo",
        run_name="run_testgym",
        telemetry_path=tmp_path / "runs" / "edge.jsonl",
    )


def _config_compute(tmp_path: Path) -> TrainConfig:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"fix","label":"1"}\n', encoding="utf-8")
    return TrainConfig(
        compute=LocalSlimeCompute(
            slime_root=tmp_path / "slime",
            megatron_root=tmp_path / "Megatron-LM",
            hf_checkpoint=tmp_path / "hf",
            ref_load=tmp_path / "ref",
            model_script="qwen2.5-3B.sh",
            repo=tmp_path / "repo",
        ),
        dataset=PromptJsonlDataset(prompts),
        recipe=CodingRecipe(batch_size=1, n_samples=1),
        run_name="run_legacy",
        telemetry_path=tmp_path / "runs" / "legacy.jsonl",
    )


def test_modal_shaped_build_dry_run(tmp_path: Path) -> None:
    cfg = _config_modal(tmp_path)
    run = cfg.launch(dry_run=True)

    assert run.dry_run is True
    assert run.training_run_id == "run_testgym"
    assert run.run_id == run.training_run_id
    joined = " ".join(run.command)
    assert "daytona_gym.adapters.slime.generate_dogfood.generate" in joined
    assert "--working-dir=" in joined
    assert str((tmp_path / "slime").resolve()) in joined
    assert "DAYTONA_API_KEY" not in run.runtime_env["env_vars"]
    assert run.runtime_env["env_vars"]["DAYTONA_MAX_CONCURRENCY"] == "4"


def test_compute_escape_hatch_still_works(tmp_path: Path) -> None:
    run = _config_compute(tmp_path).build()
    assert run.run_id == "run_legacy"
    assert "qwen2.5-3B.sh" in " ".join(run.command) or "MODEL_ARGS" in " ".join(
        run.command
    )


def test_validate_missing_paths_on_real_launch(tmp_path: Path) -> None:
    cfg = _config_modal(tmp_path)
    with pytest.raises(DaytonaError) as caught:
        cfg.launch(dry_run=False)
    assert caught.value.code == ErrorCode.USER_CODE_ERROR
    assert "missing paths" in caught.value.message


def test_requires_model_or_compute(tmp_path: Path) -> None:
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DaytonaError) as caught:
        TrainConfig(
            dataset=PromptJsonlDataset(prompts),
            recipe=CodingRecipe(),
        )
    assert caught.value.code == ErrorCode.USER_CODE_ERROR
