"""CPU contract tests for the Gym SDK build / dry_run path."""

from __future__ import annotations

from pathlib import Path

import pytest

from daytona_gym import (
    CodingRecipe,
    LocalSlimeCompute,
    PromptJsonlDataset,
    TrainConfig,
)
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def _config(tmp_path: Path, *, dataset: Path | None = None) -> TrainConfig:
    prompts = dataset or (tmp_path / "prompts.jsonl")
    if not prompts.exists():
        prompts.write_text('{"prompt":"fix","label":"1"}\n', encoding="utf-8")
    return TrainConfig(
        compute=LocalSlimeCompute(
            slime_root=tmp_path / "slime",
            megatron_root=tmp_path / "Megatron-LM",
            hf_checkpoint=tmp_path / "hf",
            ref_load=tmp_path / "ref",
            model_script="qwen2.5-3B.sh",
            repo=tmp_path / "repo",
            sglang_mem_fraction=0.5,
        ),
        dataset=PromptJsonlDataset(prompts),
        recipe=CodingRecipe(
            batch_size=2,
            n_samples=2,
            num_rollout=3,
            max_concurrency=4,
            allow_aborted=True,
            max_tools_per_turn=3,
        ),
        run_name="run_testgym",
        telemetry_path=tmp_path / "runs" / "edge.jsonl",
    )


def test_build_dry_run_includes_adapters_and_daytona_env(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run = cfg.launch(dry_run=True)

    assert run.dry_run is True
    assert run.run_id == "run_testgym"
    assert run.telemetry_path.endswith("edge.jsonl")
    joined = " ".join(run.command)
    assert "daytona_gym.adapters.slime.generate_dogfood.generate" in joined
    assert "daytona_gym.adapters.slime.reward.reward" in joined
    assert "--working-dir=" in joined
    assert str((tmp_path / "slime").resolve()) in joined
    assert str(tmp_path / "prompts.jsonl") in joined or "prompts.jsonl" in joined
    assert "--rollout-batch-size" in run.command
    assert "2" in run.command
    assert "--num-rollout" in run.command

    env_vars = run.runtime_env["env_vars"]
    assert env_vars["DAYTONA_TELEMETRY_PATH"] == str(
        (tmp_path / "runs" / "edge.jsonl").resolve()
    )
    assert env_vars["DAYTONA_RUN_ID"] == "run_testgym"
    assert env_vars["DAYTONA_MAX_CONCURRENCY"] == "4"
    assert env_vars["DAYTONA_ALLOW_ABORTED"] == "1"
    assert env_vars["DAYTONA_MAX_TOOLS_PER_TURN"] == "3"
    assert "DAYTONA_API_KEY_FILE" in env_vars
    assert "DAYTONA_API_KEY" not in env_vars
    assert "DAYTONA_API_KEY" not in run.env
    assert "dg stats" in run.inspect_hint


def test_build_matches_launch_dry_run(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    a = cfg.build()
    b = cfg.launch(dry_run=True)
    assert a.run_id == b.run_id
    assert a.command == b.command
    assert a.runtime_env == b.runtime_env


def test_validate_missing_paths_on_real_launch(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(DaytonaError) as caught:
        cfg.launch(dry_run=False)
    assert caught.value.code == ErrorCode.USER_CODE_ERROR
    assert "missing paths" in caught.value.message


def test_runtime_env_builder_refuses_api_key_injection(tmp_path: Path) -> None:
    from daytona_gym.gym.slime_command import build_runtime_env
    from daytona_gym.gym.recipe import CodingRecipe

    # Normal path is fine
    env = build_runtime_env(
        megatron_root=tmp_path / "m",
        repo=tmp_path / "r",
        key_file=tmp_path / "key",
        telemetry_path=tmp_path / "t.jsonl",
        recipe=CodingRecipe(),
        run_id="run_x",
        api_url="https://app.daytona.io/api",
    )
    assert "DAYTONA_API_KEY" not in env["env_vars"]
