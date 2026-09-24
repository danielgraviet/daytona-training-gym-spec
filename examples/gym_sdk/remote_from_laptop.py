#!/usr/bin/env python3
"""Launch a Gym job on a remote BYO GPU from your laptop.

Prefer resolving RunPod SSH for you (no hand-copy of IP/port):

  export DAYTONA_API_KEY='...'
  export RUNPOD_API_KEY='...'
  export RUNPOD_POD_ID=o2waqhc1bu3o5y          # from the pod page
  export DAYTONA_GYM_SSH_IDENTITY=~/.ssh/id_ed25519

  python examples/gym_sdk/remote_from_laptop.py --launch

Or pass host manually (homelab / any OpenSSH box):

  python examples/gym_sdk/remote_from_laptop.py \\
    --host root@1.2.3.4 --ssh-port 22 -i ~/.ssh/id_ed25519 --launch

Requires **direct TCP SSH** (SCP). The ``ssh.runpod.io`` proxy is not enough.
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
    runpod_worker,
)
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--pod",
        default=None,
        help="RunPod pod id (or RUNPOD_POD_ID) — resolves public IP + SSH port",
    )
    parser.add_argument("--host", default=None, help="Manual SSH host user@ip")
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

    remote_repo = args.remote_repo or "/root/daytona-training-gym-spec"
    if args.host:
        worker = SshWorker(
            host=args.host,
            port=args.ssh_port,
            identity=args.identity,
            remote_repo=remote_repo,
            pull=not args.no_pull,
        )
    else:
        try:
            worker = runpod_worker(
                args.pod,
                identity=args.identity,
                remote_repo=remote_repo,
                pull=not args.no_pull,
            )
        except DaytonaError:
            try:
                worker = SshWorker.from_env()
            except DaytonaError as exc:
                print(
                    "Need either --pod / RUNPOD_POD_ID+RUNPOD_API_KEY, "
                    "or --host / DAYTONA_GYM_SSH.\n"
                    f"({exc.message})",
                    file=sys.stderr,
                )
                return 2
            if args.ssh_port is not None:
                worker.port = args.ssh_port
            if args.identity:
                worker.identity = args.identity
            worker.remote_repo = remote_repo
            worker.pull = not args.no_pull

    print(f"worker={worker.host}" + (f":{worker.port}" if worker.port else ""), flush=True)

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
