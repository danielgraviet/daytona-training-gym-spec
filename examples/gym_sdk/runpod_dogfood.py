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

  # 4) dry-run / launch (launch auto-opens live dashboard URL; Ctrl+C when done)
  cd /root/daytona-training-gym-spec
  python examples/gym_sdk/runpod_dogfood.py
  python examples/gym_sdk/runpod_dogfood.py --launch

  # 5) or inspect later
  dg stats runs/<run_id>.jsonl
  dg dash
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


def print_run(run: TrainingRun, *, verbose: bool = False) -> None:
    """Short summary by default; full Ray argv only with ``verbose``."""
    status = "ok" if run.returncode in (0, None) and not run.dry_run else (
        "dry-run" if run.dry_run else f"exit={run.returncode}"
    )
    print(f"run={run.training_run_id}  {status}")
    print(f"telemetry={run.telemetry_path}")
    if run.dashboard_url:
        print(f"dashboard={run.dashboard_url}")
    else:
        print(f"inspect={run.inspect_hint}")
    env_vars = run.runtime_env.get("env_vars", {})
    if "DAYTONA_API_KEY" in env_vars:
        print("BUG: DAYTONA_API_KEY leaked into Ray runtime_env", file=sys.stderr)
    if verbose or run.dry_run:
        print("runtime_env keys:", sorted(env_vars.keys()))
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
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-start the live dashboard after launch",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit immediately after launch (dashboard would stop with the process)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full Ray command / runtime_env (noisy)",
    )
    args = parser.parse_args(argv)

    config = build_config()
    try:
        run = config.launch(
            dry_run=not args.launch,
            skip_preflight=args.skip_preflight,
            open=False if args.no_open or not args.launch else True,
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2

    print_run(run, verbose=args.verbose or not args.launch)
    if run.dry_run:
        print("\nRe-run with --launch when ready on the GPU box.")
        return 0
    # Keep process alive so the tunnel stays up (unless --no-wait).
    if run.dashboard_url and not args.no_wait:
        run.wait_dashboard()
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
