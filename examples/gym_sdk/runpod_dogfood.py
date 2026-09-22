#!/usr/bin/env python3
"""RunPod dogfood — Modal-shaped Gym SDK on slimerl/slime:latest.

Prerequisites: fresh pod from ``slimerl/slime:latest`` + Daytona API key.

  # 1) clone + install
  cd /root
  git clone https://github.com/danielgraviet/daytona-training-gym-spec.git
  cd /root/daytona-training-gym-spec
  pip install -e .

  # 2) credentials
  export DAYTONA_API_KEY='...'

  # 3) model (skip if already converted)
  hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct
  cd /root/slime
  source scripts/models/qwen2.5-3B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \\
    ${MODEL_ARGS[@]} \\
    --hf-checkpoint /root/Qwen2.5-3B-Instruct \\
    --save /root/Qwen2.5-3B-Instruct_torch_dist

  # 4) dry-run / launch
  cd /root/daytona-training-gym-spec
  python examples/gym_sdk/runpod_dogfood.py
  python examples/gym_sdk/runpod_dogfood.py --launch

  # 5) inspect
  dg stats runs/<run_id>.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daytona_gym import (
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
    TrainingRun,
)
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]


def build_config() -> TrainConfig:
    return TrainConfig(
        model=Qwen25_3B(),
        dataset=PromptJsonlDataset(
            REPO / "examples" / "coding_dogfood" / "prompts" / "coding_one.jsonl",
        ),
        recipe=Qwen25_3B_Recipe(),
        repo=REPO,
    )


def print_run(run: TrainingRun) -> None:
    print(f"training_run_id={run.training_run_id}")
    print(f"telemetry={run.telemetry_path}")
    print(f"dry_run={run.dry_run}")
    print(f"returncode={run.returncode}")
    print(f"inspect: {run.inspect_hint}")
    env_vars = run.runtime_env.get("env_vars", {})
    print("runtime_env keys:", sorted(env_vars.keys()))
    if "DAYTONA_API_KEY" in env_vars:
        print("BUG: DAYTONA_API_KEY leaked into Ray runtime_env", file=sys.stderr)
    print("command:")
    print(" ", " ".join(run.command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Execute on this GPU host (default is dry-run only)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Daytona sandbox preflight (debug only)",
    )
    args = parser.parse_args(argv)

    config = build_config()
    try:
        run = config.launch(
            dry_run=not args.launch,
            skip_preflight=args.skip_preflight,
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2

    print_run(run)
    if run.dry_run:
        print("\nRe-run with --launch when ready on the GPU box.")
        return 0
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
