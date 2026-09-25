#!/usr/bin/env python3
"""Canonical Modal-shaped entry — the only file a human writes.

Usage::

    uv run examples/gym_sdk/main.py
    # equivalent:
    dg run examples/gym_sdk/main.py

Requires ``.env`` with ``DAYTONA_API_KEY``, ``RUNPOD_*``, ``DAYTONA_GYM_SSH_IDENTITY``.
"""

from __future__ import annotations

from daytona_gym import (
    HuggingFaceDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
    runpod_worker,
)
from daytona_gym.envfile import load_dotenv

load_dotenv()

config = TrainConfig(
    model=Qwen25_3B(),
    dataset=HuggingFaceDataset(
        hf_repo="openai/gsm8k",
        hf_config="main",
        hf_split="train[:32]",
        input_column="question",
        output_column="answer",
        prompt_template="Solve step by step:\n{input}",
    ),
    recipe=Qwen25_3B_Recipe(),
)

if __name__ == "__main__":
    run = config.launch(worker=runpod_worker(), detach=True)
    print(run.training_run_id)
    print(run.dashboard_url)  # always http://127.0.0.1:<port>/run/<id>
    run.result()
