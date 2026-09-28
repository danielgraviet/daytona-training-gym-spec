#!/usr/bin/env python3
"""Repo-scale pack dogfood — multi-file calc bug + optional Daytona snapshot.

Compare provision p95 with vs without ``recipe.snapshot=...`` via ``dg stats``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from daytona_gym.envfile import load_dotenv

load_dotenv()

from daytona_gym import (  # noqa: E402
    HarborDataset,
    Qwen25_05B,
    Qwen25_05B_Recipe,
    TrainConfig,
    vast_worker,
)
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "examples" / "gym_sdk" / "repo_pack"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument(
        "--snapshot",
        default=os.environ.get("DAYTONA_SNAPSHOT") or None,
        help="Daytona snapshot name/id (omit for cold image provision)",
    )
    args = parser.parse_args(argv)

    worker = vast_worker(remote_repo="/root/daytona-training-gym-spec", pull=True)
    recipe = Qwen25_05B_Recipe(
        batch_size=1,
        n_samples=1,
        num_rollout=1,
        max_turns=6,
        offload_train=False,
        offload_rollout=False,
        snapshot=args.snapshot,
    )
    config = TrainConfig(
        model=Qwen25_05B(sglang_mem_fraction=0.15),
        dataset=HarborDataset(path=PACK, train_size=1, shuffle_tasks=False),
        recipe=recipe,
        repo=REPO,
        gpu_cost_per_hour=float(os.environ.get("GPU_COST_PER_HOUR", "0.4")) or None,
    )
    try:
        run = config.launch(
            worker=worker,
            dry_run=not args.launch,
            detach=args.detach,
        )
    except DaytonaError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 2
    print(f"run={run.training_run_id} snapshot={args.snapshot!r}")
    if run.dashboard_url:
        print(f"dashboard={run.dashboard_url}")
    if not args.launch:
        print("Re-run with --launch (optionally --snapshot NAME).")
        return 0
    if run.detached:
        run.result()
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
