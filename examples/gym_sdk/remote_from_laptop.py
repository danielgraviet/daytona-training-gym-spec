#!/usr/bin/env python3
"""Launch a Gym job on a remote BYO GPU from your laptop.

Requires **direct TCP SSH** to the worker (OpenSSH with SCP), e.g. RunPod
Connect → SSH over exposed TCP ``root@PUBLIC_IP -p PORT`` — not ``ssh.runpod.io``.

  export DAYTONA_API_KEY='...'
  export DAYTONA_GYM_SSH=root@64.x.x.x
  export DAYTONA_GYM_SSH_PORT=12713
  export DAYTONA_GYM_SSH_IDENTITY=~/.ssh/id_ed25519

  # worker already has slime:latest layout + this repo cloned
  python examples/gym_sdk/remote_from_laptop.py
  python examples/gym_sdk/remote_from_laptop.py --launch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daytona_gym import (
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    SshWorker,
    TrainConfig,
)
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--host", default=None, help="Override DAYTONA_GYM_SSH")
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("-i", "--identity", default=None)
    parser.add_argument(
        "--remote-repo",
        default=None,
        help="Repo path on the worker (default /root/daytona-training-gym-spec)",
    )
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    if args.host:
        worker = SshWorker(
            host=args.host,
            port=args.ssh_port,
            identity=args.identity,
            remote_repo=args.remote_repo or "/root/daytona-training-gym-spec",
            pull=not args.no_pull,
        )
    else:
        worker = SshWorker.from_env()
        if args.ssh_port is not None:
            worker.port = args.ssh_port
        if args.identity:
            worker.identity = args.identity
        if args.remote_repo:
            worker.remote_repo = args.remote_repo
        worker.pull = not args.no_pull

    config = TrainConfig(
        model=Qwen25_3B(),
        dataset=PromptJsonlDataset(
            REPO / "examples" / "coding_dogfood" / "prompts" / "coding_one.jsonl"
        ),
        recipe=Qwen25_3B_Recipe(),
        repo=REPO,
    )

    try:
        run = config.launch(
            worker=worker,
            dry_run=not args.launch,
            skip_preflight=args.skip_preflight,
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2

    print(f"run={run.training_run_id}  exit={run.returncode}")
    if run.dashboard_url:
        print(f"dashboard={run.dashboard_url}")
    else:
        print(f"inspect={run.inspect_hint}")
    if not args.launch:
        print("\nRe-run with --launch when the worker SSH is ready.")
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
