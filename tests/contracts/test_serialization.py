from __future__ import annotations

import json

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def test_adapter_metadata_is_json_serializable() -> None:
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [
            tool_turn("write_file", {"path": "a.txt", "content": "alpha"}),
            final_turn("wrote file"),
        ]
    )
    sample = FakeSlimeSample(prompt="write", index=5, rollout_id=11)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    payload = json.dumps(sample.metadata)
    loaded = json.loads(payload)
    assert loaded["daytona"]["run_id"] == "run_test"
    assert loaded["daytona"]["rollout_id"] == "rollout_11"
    assert loaded["daytona"]["events"][0]["type"] == "generation"
    assert isinstance(loaded["daytona"]["events"][0]["text_chars"], int)
