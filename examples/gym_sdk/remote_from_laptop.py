#!/usr/bin/env python3
"""Legacy argparse launcher — prefer ``examples/gym_sdk/main.py`` + ``uv run`` / ``dg run``.

Kept as a thin escape hatch for ``--create-pod`` / explicit ``--host`` flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daytona_gym.envfile import load_dotenv

load_dotenv()

from daytona_gym import (  # noqa: E402
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    SshWorker,
    TrainConfig,
    runpod_worker,
)
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]


def _build_worker(args: argparse.Namespace):
    remote_repo = args.remote_repo or "/root/daytona-training-gym-spec"
    if args.host:
        return SshWorker(
            host=args.host,
            port=args.ssh_port,
            identity=args.identity,
            remote_repo=remote_repo,
            pull=not args.no_pull,
        )
    return runpod_worker(
        args.pod,
        identity=args.identity,
        remote_repo=remote_repo,
        pull=not args.no_pull,
        create=bool(args.create_pod),
    )


def main() -> int:
    print(
        "note: prefer `uv run examples/gym_sdk/main.py` (no argparse).",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description="Legacy remote launcher — use examples/gym_sdk/main.py instead"
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("-i", "--identity", default=None)
    parser.add_argument("--pod", default=None)
    parser.add_argument("--create-pod", action="store_true")
    parser.add_argument("--remote-repo", default=None)
    parser.add_argument("--no-pull", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        model=Qwen25_3B(),
        dataset=PromptJsonlDataset(
            REPO / "examples/coding_dogfood/prompts/coding_one.jsonl"
        ),
        recipe=Qwen25_3B_Recipe(),
        repo=REPO,
    )
    try:
        worker = _build_worker(args)
    except DaytonaError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 2

    run = config.launch(
        worker=worker,
        dry_run=not args.launch,
        detach=args.detach,
    )
    print(run.training_run_id)
    if run.dashboard_url:
        print(run.dashboard_url)
    if args.launch and args.detach and not args.no_wait:
        run.result()
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
