#!/usr/bin/env python3
"""Gym SDK quickstart — dry-run by default; pass --launch on a Slime GPU host.

  python examples/gym_sdk/quickstart.py
  DAYTONA_API_KEY=... python examples/gym_sdk/quickstart.py --launch
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from daytona_gym import (
    CodingRecipe,
    LocalSlimeCompute,
    PromptJsonlDataset,
    TrainConfig,
)

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Daytona Gym SDK coding dogfood")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Run on this host (requires Slime/Megatron/Ray + DAYTONA_API_KEY)",
    )
    parser.add_argument(
        "--slime-root",
        default=os.environ.get("SLIME_ROOT", "/root/slime"),
    )
    parser.add_argument(
        "--megatron-root",
        default=os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM"),
    )
    parser.add_argument(
        "--hf-checkpoint",
        default=os.environ.get("HF_CHECKPOINT", "/root/Qwen2.5-3B-Instruct/"),
    )
    parser.add_argument(
        "--ref-load",
        default=os.environ.get("REF_LOAD", "/root/Qwen2.5-3B-Instruct_torch_dist/"),
    )
    parser.add_argument("--model-script", default=os.environ.get("MODEL_SCRIPT", "qwen2.5-3B.sh"))
    parser.add_argument(
        "--prompt-data",
        default=str(REPO / "examples/coding_dogfood/prompts/coding_one.jsonl"),
    )
    args = parser.parse_args()

    config = TrainConfig(
        compute=LocalSlimeCompute(
            slime_root=args.slime_root,
            megatron_root=args.megatron_root,
            hf_checkpoint=args.hf_checkpoint,
            ref_load=args.ref_load,
            model_script=args.model_script,
            repo=REPO,
        ),
        dataset=PromptJsonlDataset(args.prompt_data),
        recipe=CodingRecipe(batch_size=1, n_samples=1, num_rollout=1),
    )

    if not args.launch:
        run = config.launch(dry_run=True)
        print(f"run_id={run.run_id}")
        print(f"telemetry={run.telemetry_path}")
        print(f"inspect: {run.inspect_hint}")
        print("command (dry-run):")
        print(" ", " ".join(run.command[:12]), "...")
        print("Pass --launch on a Slime GPU host to execute.")
        return 0

    run = config.launch(dry_run=False)
    print(f"run_id={run.run_id} returncode={run.returncode}")
    print(run.inspect_hint)
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
