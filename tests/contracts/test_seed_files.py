from __future__ import annotations

from pathlib import Path

from daytona_gym.adapters.slime import generate
from daytona_gym.adapters.slime._coding_seed import CODING_SEED_FILES
from daytona_gym.adapters.slime.generate_dogfood import generate as dogfood_generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.inspect import main as inspect_main
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_seed_files_written_before_generate(tmp_path: Path, capsys) -> None:
    path = tmp_path / "seed.jsonl"
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_tests", {"command": "python test_broken.py"}),
            tool_turn(
                "write_file",
                {
                    "path": "broken.py",
                    "content": "def add(a, b):\n    return a + b\n",
                },
            ),
            tool_turn("run_tests", {"command": "python test_broken.py"}),
            final_turn("fixed"),
        ]
    )
    sample = FakeSlimeSample(prompt="fix", index=7)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
        daytona_seed_files=dict(CODING_SEED_FILES),
        daytona_run_id="seed_test",
    )
    try:
        await generate(args, sample, {})
        args.daytona_telemetry_store.flush()
    finally:
        args.daytona_telemetry_store.close()

    files = runtime.files_for(runtime.created_ids[0])
    assert "broken.py" in files
    assert "test_broken.py" in files
    assert "return a + b" in files["broken.py"]
    assert sample.metadata["daytona"]["sandbox_id"]
    assert sample.reward == 1.0

    assert inspect_main([str(path), "--rollout", "rollout_7"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.seed" in out
    assert "tool.run_tests" in out
    assert "tool.write_file" in out


async def test_generate_dogfood_defaults_seed(tmp_path: Path) -> None:
    path = tmp_path / "dogfood.jsonl"
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            final_turn("done"),
        ]
    )
    sample = FakeSlimeSample(prompt="fix", index=3)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
    )
    if hasattr(args, "daytona_seed_files"):
        delattr(args, "daytona_seed_files")
    try:
        await dogfood_generate(args, sample, {})
        args.daytona_telemetry_store.flush()
    finally:
        args.daytona_telemetry_store.close()

    files = runtime.files_for(runtime.created_ids[0])
    assert files["broken.py"].startswith("def add")
    assert "test_broken.py" in files
    # Bootstrap run_tests fails on seeded broken.py → reward 0.0
    assert sample.reward == 0.0
    assert args.daytona_bootstrap_run_tests == "python test_broken.py"


async def test_bootstrap_run_tests_emits_tool_span(tmp_path: Path, capsys) -> None:
    path = tmp_path / "boot.jsonl"
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("give up")])
    sample = FakeSlimeSample(prompt="fix", index=9)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
        daytona_seed_files=dict(CODING_SEED_FILES),
        daytona_bootstrap_run_tests="python test_broken.py",
        daytona_run_id="boot_test",
    )
    try:
        await generate(args, sample, {})
        args.daytona_telemetry_store.flush()
    finally:
        args.daytona_telemetry_store.close()

    assert sample.reward == 0.0
    assert inspect_main([str(path), "--rollout", "rollout_9"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.seed" in out
    assert "tool.run_tests" in out


async def test_bootstrap_from_ray_runtime_env(tmp_path: Path, monkeypatch, capsys) -> None:
    """Bootstrap must work when only Ray runtime_env vars are set (no args)."""
    monkeypatch.setenv("DAYTONA_BOOTSTRAP_RUN_TESTS", "1")
    monkeypatch.setenv("DAYTONA_BOOTSTRAP_RUN_TESTS_CMD", "python test_broken.py")
    monkeypatch.setenv("DAYTONA_SEED_CODING", "1")

    path = tmp_path / "env_boot.jsonl"
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator([final_turn("give up")])
    sample = FakeSlimeSample(prompt="fix", index=11)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
        daytona_run_id="env_boot",
    )
    if hasattr(args, "daytona_seed_files"):
        delattr(args, "daytona_seed_files")
    if hasattr(args, "daytona_bootstrap_run_tests"):
        delattr(args, "daytona_bootstrap_run_tests")

    try:
        await generate(args, sample, {})
        args.daytona_telemetry_store.flush()
    finally:
        args.daytona_telemetry_store.close()

    files = runtime.files_for(runtime.created_ids[0])
    assert "broken.py" in files
    assert sample.reward == 0.0
    assert inspect_main([str(path), "--rollout", "rollout_11"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.seed" in out
    assert "tool.run_tests" in out
    assert "bootstrap='python test_broken.py'" in out
