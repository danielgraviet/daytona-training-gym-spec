"""Run two fake rollouts and write inspectable JSONL under ./runs/."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn

OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "inspect-telemetry.jsonl"


async def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    runtime = FakeEnvironmentRuntime()
    args = make_args(
        runtime=runtime,
        generator=ScriptedGenerator(
            [
                tool_turn("write_file", {"path": "note.txt", "content": "hello"}),
                tool_turn("run_command", {"command": "echo hello"}),
                final_turn("done"),
            ]
        ),
        daytona_telemetry_path=str(OUTPUT),
        daytona_run_id="run_inspect",
        daytona_worker_id="local-dev",
        daytona_training_step=1,
        daytona_rollout_batch_id="batch-inspect",
        daytona_reward_function=lambda _args, _sample: 1.0,
    )
    first = FakeSlimeSample(prompt="write a note", index=1, rollout_id=1)
    await generate(args, first, {"temperature": 0.0})

    args.daytona_generator = ScriptedGenerator([final_turn("second")])
    args.daytona_training_step = 2
    second = FakeSlimeSample(prompt="follow up", index=2, rollout_id=2)
    await generate(args, second, {"temperature": 0.0})

    args.daytona_telemetry_store.flush()
    args.daytona_telemetry_store.close()

    lines = OUTPUT.read_text(encoding="utf-8").splitlines()
    print(f"wrote {len(lines)} records to {OUTPUT}")
    print(f"rollout_1 status={first.status} reward={first.reward}")
    print(f"rollout_2 status={second.status} reward={second.reward}")


if __name__ == "__main__":
    asyncio.run(main())
