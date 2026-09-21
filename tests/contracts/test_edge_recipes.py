from __future__ import annotations

import pytest

from daytona_gym.adapters.slime._coding_seed import resolve_seed_profile
from daytona_gym.adapters.slime.generate_dogfood import generate as dogfood_generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


def test_seed_profiles() -> None:
    basic = resolve_seed_profile("basic")
    multi = resolve_seed_profile("multifile")
    assert "broken.py" in basic
    assert "util.py" in multi
    assert "mul" in multi["util.py"]
    with pytest.raises(ValueError):
        resolve_seed_profile("nope")


async def test_dogfood_honors_seed_profile_and_timeouts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "edge.jsonl"
    monkeypatch.setenv("DAYTONA_SEED_PROFILE", "multifile")
    monkeypatch.setenv("DAYTONA_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("DAYTONA_TOOL_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("DAYTONA_BOOTSTRAP_RUN_TESTS", "1")
    monkeypatch.setenv("DAYTONA_TELEMETRY_PATH", str(path))

    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn(
                "write_file",
                {"path": "util.py", "content": "def mul(a, b):\n    return a * b\n"},
            ),
            tool_turn("run_tests", {"command": "python test_broken.py"}),
            final_turn("fixed"),
        ]
    )
    sample = FakeSlimeSample(prompt="fix", index=0)
    args = make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_path=str(path),
    )
    # Clear seed so dogfood resolves from DAYTONA_SEED_PROFILE.
    args.daytona_seed_files = None
    try:
        await dogfood_generate(args, sample, {})
        args.daytona_telemetry_store.flush()
    finally:
        args.daytona_telemetry_store.close()

    assert args.daytona_timeout_seconds == 30.0
    assert args.daytona_tool_timeout_seconds == 10.0
    assert "util.py" in (args.daytona_seed_files or {})
