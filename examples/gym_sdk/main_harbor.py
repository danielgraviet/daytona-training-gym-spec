#!/usr/bin/env python3
"""Harbor-shaped coding dogfood — tasks carry seed files + tests.

Uses a local Harbor-like task pack under ``examples/gym_sdk/harbor_tasks/`` so
dogfood works without the Harbor CLI. Swap in::

    HarborDataset(dataset_name=\"your-org/your-pack\")

once ``harbor datasets download`` is available on the worker.
"""

from __future__ import annotations

from pathlib import Path

from daytona_gym import (
    HarborDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
    runpod_worker,
)
from daytona_gym.envfile import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parents[2]

config = TrainConfig(
    model=Qwen25_3B(),
    dataset=HarborDataset(
        path=REPO / "examples/gym_sdk/harbor_tasks",
        train_size=2,
        shuffle_seed=0,
    ),
    recipe=Qwen25_3B_Recipe(),
    repo=REPO,
)

if __name__ == "__main__":
    run = config.launch(worker=runpod_worker(), detach=True)
    print(run.training_run_id)
    print(run.dashboard_url)
    run.result()
