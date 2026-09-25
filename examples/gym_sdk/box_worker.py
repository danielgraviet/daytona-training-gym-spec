"""Train on your own GPU box from the worker image — no git clone needed.

Pipe this file into the container (the image already has Slime + daytona_gym):

    docker run --gpus all --rm -i --ipc=host --shm-size=16g \
      -v ~/gym-models:/models -v ~/gym-work:/workspace \
      --env-file ~/.daytona-gym.env \
      ghcr.io/danielgraviet/daytona-gym-worker:latest python - < examples/gym_sdk/box_worker.py

``~/.daytona-gym.env`` (chmod 600) holds DAYTONA_API_KEY, DAYTONA_GYM_INGEST_URL
and DAYTONA_GYM_INGEST_TOKEN (plus HF_TOKEN if needed). Models land in
~/gym-models and are reused next time; runs/ lands in ~/gym-work.

Sized for a 24 GB card (RTX 3090 / 4090): Qwen2.5-0.5B, colocated.
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
    model=Qwen25_05B(),
    dataset=PromptJsonlDataset(data),
    recipe=Qwen25_05B_Recipe(
        batch_size=4,
        n_samples=2,
        num_rollout=4,
        max_turns=6,
        timeout_seconds=180,
        tool_timeout_seconds=60,
    ),
    gpu_cost_per_hour=float(os.environ.get("GPU_COST_PER_HOUR", "0.0")) or None,
)

if __name__ == "__main__":
    run = config.launch(open=True)
    print(run.training_run_id)
    print(run.dashboard_url or run.inspect_hint)
