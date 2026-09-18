"""CPU dry-run of the dogfood path: SGLang fake router + sandbox + JSONL + inspect."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daytona_gym.adapters.slime import generate
from daytona_gym.adapters.slime.reward import reward as slime_reward
from daytona_gym.adapters.slime.sglang_generator import SGLangRouterGenerator
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.telemetry.inspect import main as inspect_main
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn

OUTPUT = ROOT / "runs" / "dogfood-dryrun.jsonl"


async def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    turns = [
        tool_turn("run_tests", {"command": "python test_broken.py"}),
        tool_turn(
            "write_file",
            {"path": "broken.py", "content": "def add(a, b):\n    return a + b\n"},
        ),
        tool_turn("run_tests", {"command": "python test_broken.py"}),
        final_turn("fixed"),
    ]
    # Scripted-style responses via fake SGLang router so we exercise the
    # production default generator wiring.
    queue = list(turns)

    async def fake_post(url: str, payload: dict) -> dict:
        assert url.endswith("/generate")
        text = queue.pop(0)
        return {
            "text": text,
            "meta_info": {
                "id": f"req-{len(turns) - len(queue)}",
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.01, ord(c)] for c in text],
            },
        }

    from daytona_gym.adapters.slime._coding_seed import CODING_SEED_FILES

    runtime = FakeEnvironmentRuntime()
    # Make second pytest succeed by scripting write then tests via fake files.
    args = make_args(
        runtime=runtime,
        generator=None,
        daytona_telemetry_path=str(OUTPUT),
        daytona_run_id="dogfood_dryrun",
        daytona_project_id="coding-rl",
        daytona_worker_id="local-dev",
        daytona_training_step=1,
        daytona_seed_files=dict(CODING_SEED_FILES),
    )
    args.daytona_generator = None
    args.sglang_router_ip = "127.0.0.1"
    args.sglang_router_port = 30000
    args.daytona_sglang_post_fn = fake_post

    sample = FakeSlimeSample(prompt="Fix the failing tests", index=0, rollout_id=0)
    await generate(args, sample, {"temperature": 0.0})
    rm = await slime_reward(args, sample)

    args.daytona_telemetry_store.flush()
    args.daytona_telemetry_store.close()

    print(f"sample status={sample.status} reward={sample.reward} rm={rm}")
    print(f"response_length={sample.response_length} loss_mask_len={len(sample.loss_mask or [])}")
    print(f"logprobs={sample.rollout_log_probs is not None} generator={type(args.daytona_generator).__name__}")
    print(f"wrote {OUTPUT}")
    raise SystemExit(inspect_main([str(OUTPUT), "--rollout", "rollout_0"]))


if __name__ == "__main__":
    asyncio.run(main())
