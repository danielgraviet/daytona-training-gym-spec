from __future__ import annotations

from pathlib import Path

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry.inspect import main as inspect_main
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_inspect_prints_timeline(tmp_path: Path, capsys) -> None:
    path = tmp_path / "dogfood.jsonl"
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("run_tests", {"command": "pytest"}),
            final_turn("fixed"),
        ]
    )
    sample = FakeSlimeSample(prompt="fix", index=1)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
    )
    try:
        await generate(args, sample, {})
        args.daytona_telemetry_store.flush()
    finally:
        args.daytona_telemetry_store.close()

    assert path.exists()
    assert inspect_main([str(path), "--rollout", "rollout_1"]) == 0
    out = capsys.readouterr().out
    assert "rollout_1" in out
    assert "provision" in out
    assert "generate" in out
    assert "run_tests" in out
    assert "reward" in out


def test_inspect_defaults_and_help(capsys) -> None:
    from daytona_gym.cli import main as cli_main

    assert cli_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "dg" in out
    assert "dg ls" in out
    assert "dg stats" in out


async def test_dg_ls_and_stats(tmp_path: Path, capsys) -> None:
    from daytona_gym.cli import main as cli_main

    path = tmp_path / "dogfood.jsonl"
    runtime = FakeEnvironmentRuntime()
    for index in (0, 1):
        generator = ScriptedGenerator(
            [
                tool_turn("run_tests", {"command": "pytest"}),
                final_turn("fixed"),
            ]
        )
        sample = FakeSlimeSample(prompt="fix", index=index)
        args = make_args(
            runtime=runtime,
            generator=generator,
            daytona_telemetry_path=str(path),
        )
        try:
            await generate(args, sample, {})
            args.daytona_telemetry_store.flush()
        finally:
            args.daytona_telemetry_store.close()

    assert cli_main(["ls", str(path)]) == 0
    ls_out = capsys.readouterr().out
    assert "2 rollouts" in ls_out
    assert "rollout_0" in ls_out
    assert "rollout_1" in ls_out
    assert "reward=" in ls_out

    assert cli_main(["stats", str(path)]) == 0
    stats_out = capsys.readouterr().out
    assert "n=2" in stats_out
    assert "status" in stats_out
    assert "reward" in stats_out
    assert "tools" in stats_out
