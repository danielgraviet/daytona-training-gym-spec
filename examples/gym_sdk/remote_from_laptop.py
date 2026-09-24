#!/usr/bin/env python3
"""Launch a Gym job on a remote BYO GPU from your **laptop** (Mac / coding agent).

This is the path agents should use — you should not SSH in and paste commands.

RunPod (recommended)::

  export DAYTONA_API_KEY='...'
  export RUNPOD_API_KEY='...'
  export RUNPOD_POD_ID=yourpodid
  export DAYTONA_GYM_SSH_IDENTITY=~/.ssh/id_ed25519

  python examples/gym_sdk/remote_from_laptop.py --launch

Detached (return run id + dash URL; training keeps going on the pod)::

  python examples/gym_sdk/remote_from_laptop.py --launch --detach

Uses RunPod proxy SSH (``user@ssh.runpod.io``) with a PTY shell, so it works
even when the container has no ``sshd`` (e.g. slime + ``sleep infinity``).

Homelab / direct TCP SSH::

  python examples/gym_sdk/remote_from_laptop.py \\
    --host root@1.2.3.4 --ssh-port 22 -i ~/.ssh/id_ed25519 --launch

If you are already on the GPU box, you can still use::

  python examples/gym_sdk/runpod_dogfood.py --launch
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
from daytona_gym.runtime.errors import DaytonaError, ErrorCode

REPO = Path(__file__).resolve().parents[2]


def _build_worker(args: argparse.Namespace):
    remote_repo = args.remote_repo or "/root/daytona-training-gym-spec"
    errors: list[str] = []

    if args.host:
        return SshWorker(
            host=args.host,
            port=args.ssh_port,
            identity=args.identity,
            remote_repo=remote_repo,
            pull=not args.no_pull,
        )

    try:
        return runpod_worker(
            args.pod,
            identity=args.identity,
            remote_repo=remote_repo,
            pull=not args.no_pull,
        )
    except DaytonaError as exc:
        errors.append(f"runpod: [{exc.code}] {exc.message}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"runpod: {exc}")

    try:
        worker = SshWorker.from_env()
    except DaytonaError as exc:
        errors.append(f"ssh-env: {exc.message}")
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            "Could not build a worker.\n  - "
            + "\n  - ".join(errors)
            + "\n\nNeed RUNPOD_POD_ID+RUNPOD_API_KEY (laptop → RunPod proxy), "
            "or --host / DAYTONA_GYM_SSH.",
        ) from exc

    if args.ssh_port is not None:
        worker.port = args.ssh_port
    if args.identity:
        worker.identity = args.identity
    worker.remote_repo = remote_repo
    worker.pull = not args.no_pull
    return worker


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
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Return as soon as run id + dashboard URL are ready; train continues on the pod",
    )
    args = parser.parse_args(argv)

    try:
        worker = _build_worker(args)
    except DaytonaError as exc:
        print(exc.message, file=sys.stderr)
        return 2

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
            detach=args.detach,
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print(f"run={run.training_run_id}")
    if run.dashboard_url:
        print(f"dashboard={run.dashboard_url}")
    else:
        print(f"inspect={run.inspect_hint}")
    if run.detached:
        print("detached=true  (training continues on the worker)")
    elif run.returncode is not None:
        print(f"exit={run.returncode}")
    if not args.launch:
        print("\nRe-run with --launch when the worker SSH is ready.")
    return 0 if run.detached else int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
