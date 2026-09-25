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


def test_training_run_open_sets_dashboard_url(tmp_path: Path, monkeypatch) -> None:
    from daytona_gym.gym.run import TrainingRun
    from daytona_gym.telemetry.dashboard import DashboardHandle

    telemetry = tmp_path / "runs" / "run_abc.jsonl"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text("{}\n")

    class FakeHttpd:
        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    fake = DashboardHandle(
        url="https://example.trycloudflare.com/",
        local_url="http://127.0.0.1:8765/",
        runs_dir=telemetry.parent,
        shared=True,
        _httpd=FakeHttpd(),  # type: ignore[arg-type]
        _thread=__import__("threading").Thread(target=lambda: None),
        _tunnel=None,
    )

    def fake_start(**kwargs):  # noqa: ANN003
        return fake

    monkeypatch.setattr(
        "daytona_gym.telemetry.dashboard.start_dashboard",
        fake_start,
    )

    run = TrainingRun(
        run_id="run_abc",
        telemetry_path=str(telemetry),
        command=["echo"],
        env={},
        runtime_env={"env_vars": {}},
        dry_run=False,
        returncode=0,
    )
    url = run.open(share=True)
    assert url == "https://example.trycloudflare.com/run/run_abc"
    assert run.dashboard_url == url
    assert "trycloudflare.com/run/run_abc" in run.inspect_hint
    run.close_dashboard()


def test_launch_open_false_skips_dashboard(tmp_path: Path, monkeypatch) -> None:
    cfg = _config_modal(tmp_path)
    # create minimal paths so validate passes, then stub execute_plan
    for sub in ("slime", "Megatron-LM", "hf", "ref", "repo"):
        (tmp_path / sub).mkdir(exist_ok=True)
    (tmp_path / "slime" / "train.py").write_text("#\n")

    from daytona_gym.gym import run as run_mod

    def fake_execute(plan, **kwargs):  # noqa: ANN003
        return run_mod.TrainingRun(
            run_id=plan.run_id,
            telemetry_path=str(plan.telemetry_path),
            command=["ray", "job", "submit"],
            env={},
            runtime_env={"env_vars": {}},
            dry_run=False,
            returncode=0,
        )

    monkeypatch.setattr("daytona_gym.gym.config.execute_plan", fake_execute)
    opened = {"n": 0}

    def boom(*a, **k):  # noqa: ANN001
        opened["n"] += 1
        raise AssertionError("open should not be called")

    monkeypatch.setattr(run_mod.TrainingRun, "open", boom)
    run = cfg.launch(dry_run=False, open=False, skip_preflight=True)
    assert run.returncode == 0
    assert opened["n"] == 0
    assert run.dashboard_url is None


@pytest.mark.parametrize(
    ("kind", "explicit", "expect_share"),
    [
        ("runpod", None, True),  # on-pod launch: loopback is unreachable → tunnel
        ("ssh", None, True),
        ("local", None, False),  # laptop: stay on loopback
        ("runpod", False, False),  # laptop-driven remote_job opts out explicitly
    ],
)
def test_open_shares_by_default_only_on_gpu_boxes(
    tmp_path: Path, monkeypatch, kind: str, explicit, expect_share: bool
) -> None:
    from types import SimpleNamespace

    from daytona_gym.gym.run import TrainingRun
    from daytona_gym.telemetry.dashboard import ServeContext

    seen: dict = {}

    def fake_start(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        return SimpleNamespace(url="https://x.trycloudflare.com/", local_url="http://127.0.0.1:3000/", stop=lambda: None)

    monkeypatch.setattr("daytona_gym.telemetry.dashboard.start_dashboard", fake_start)
    monkeypatch.setattr(
        "daytona_gym.telemetry.dashboard.detect_serve_context", lambda: ServeContext(kind)
    )
    run = TrainingRun(
        run_id="r",
        telemetry_path=str(tmp_path / "runs" / "r.jsonl"),
        command=[],
        env={},
        runtime_env={},
    )
    url = run.open(share=explicit)
    assert seen["share"] is expect_share
    assert url.startswith("https://") is expect_share
