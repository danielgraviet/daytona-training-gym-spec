"""Train on your own GPU box from the worker image — no git clone needed.

Pipe this file into the container (the image already has Slime + daytona_gym):

    docker run --gpus all --rm -i --ipc=host \
      -v daytona-gym-models:/models --env-file ~/.daytona-gym.env \
      ghcr.io/danielgraviet/daytona-gym-worker:latest python - < examples/gym_sdk/box_worker.py

``daytona-gym-models`` is a Docker named volume (created automatically) so model
weights survive between runs. Run history goes to the ingest host, so nothing
else needs to persist on the box. ``~/.daytona-gym.env`` (chmod 600) holds
DAYTONA_API_KEY, DAYTONA_GYM_INGEST_URL and DAYTONA_GYM_INGEST_TOKEN.

Sized for a 24 GB card (RTX 3090 / 4090) in a 16 GB-RAM box: Qwen2.5-0.5B,
colocated with both engines kept on the GPU (no CPU offload). Default colocated
Slime parks the idle engine in host RAM each step, which OOM-killed the
trainer on a 15 GiB box; here SGLang is capped at 20% of VRAM instead and
samples are kept short enough for their logits to fit on the GPU.
"""

import json
import os
from pathlib import Path

from daytona_gym import PromptJsonlDataset, Qwen25_05B, Qwen25_05B_Recipe, TrainConfig
from daytona_gym.adapters.slime._coding_seed import CODING_DOGFOOD_PROMPT

workdir = Path(os.environ.get("DAYTONA_GYM_WORKDIR", ".")).resolve()
data = workdir / "data" / "coding_seed.jsonl"
data.parent.mkdir(parents=True, exist_ok=True)
data.write_text(
    "".join(json.dumps({"prompt": CODING_DOGFOOD_PROMPT, "label": "fixed"}) + "\n" for _ in range(16))
)

config = TrainConfig(
    # 15% of VRAM (~3.5 GB) is enough SGLang for 0.5B; the rest is Megatron's.
    model=Qwen25_05B(sglang_mem_fraction=0.15),
    dataset=PromptJsonlDataset(data),
    recipe=Qwen25_05B_Recipe(
        batch_size=4,
        n_samples=2,
        num_rollout=4,
        # Keep each multi-turn sample ~2k tokens: its fp32 logits (152k vocab)
        # must fit next to the resident engines. Live 3090: 6 turns OOM'd at ~5k
        # tokens, 3 turns at a 2.8k-token straggler in step 3.
        max_turns=2,
        max_response_len=512,
        timeout_seconds=180,
        tool_timeout_seconds=60,
        offload_train=False,
        offload_rollout=False,
    ),
    gpu_cost_per_hour=float(os.environ.get("GPU_COST_PER_HOUR", "0.0")) or None,
)

if __name__ == "__main__":
    run = config.launch(open=True)
    print(run.training_run_id)
    print(run.dashboard_url or run.inspect_hint)
    raise SystemExit(int(run.returncode or 0))  # a failed run must not look like success
