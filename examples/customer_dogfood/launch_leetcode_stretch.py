#!/usr/bin/env python3
"""LeetCode-style tasks with real sandbox seeds (Phase 4 exit path).

Uses the local Harbor-shaped pack under ``examples/gym_sdk/harbor_tasks/``
(Two Sum + Valid Parentheses) so each row carries ``seed_files`` +
``run_tests_command`` — no silent add(a,b) fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daytona_gym.envfile import load_dotenv

load_dotenv()

from daytona_gym import (  # noqa: E402
    HarborDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
    runpod_worker,
    vast_worker,
)
from daytona_gym.adapters.slime.reward import from_trajectory  # noqa: E402
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "examples" / "gym_sdk" / "harbor_tasks"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--worker",
        choices=("runpod", "vast"),
        default="runpod",
        help="BYO GPU provider (Vast kept warm for continual Phase 4 dogfood)",
    )
    args = parser.parse_args(argv)

    if args.worker == "vast":
        worker = vast_worker(remote_repo="/root/daytona-training-gym-spec", pull=True)
    else:
        worker = runpod_worker(remote_repo="/root/daytona-training-gym-spec", pull=True)
    print(f"worker={worker.host}" + (f":{worker.port}" if worker.port else ""), flush=True)

    config = TrainConfig(
        model=Qwen25_3B(),
        dataset=HarborDataset(
            path=TASKS,
            train_size=2,
            shuffle_seed=0,
            shuffle_tasks=False,
        ),
        recipe=Qwen25_3B_Recipe(batch_size=1, n_samples=1, num_rollout=1),
        reward=from_trajectory,
        repo=REPO,
    )

    try:
        run = config.launch(
            worker=worker,
            dry_run=not args.launch,
            skip_preflight=args.skip_preflight,
            detach=args.detach,
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2

    print(f"run={run.training_run_id}")
    if run.dashboard_url:
        print(f"dashboard={run.dashboard_url}")
    if run.detached:
        print("detached=true", flush=True)
    if not args.launch:
        print("Re-run with --launch when ready.")
        return 0
    if run.detached and args.no_wait:
        return 0
    if run.detached:
        run.result()
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
