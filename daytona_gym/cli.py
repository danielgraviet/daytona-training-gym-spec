"""Short CLI entrypoints.

Examples:
  dg
  dg inspect
  dg runs/other.jsonl -r rollout_0
  python -m daytona_gym
"""

from __future__ import annotations

import sys

from daytona_gym.telemetry.inspect import main as inspect_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"inspect", "i"}:
        args = args[1:]
    elif args and args[0] in {"-h", "--help"}:
        print(
            "dg — Daytona Gym helpers\n\n"
            "  dg                 inspect runs/dogfood.jsonl (first rollout)\n"
            "  dg inspect [path]  same; path optional\n"
            "  dg i [path]        alias for inspect\n"
            "  dg path.jsonl -r rollout_0\n"
            "  dg --list-rollouts\n"
            "  dg --raw           legacy dense format\n"
        )
        return 0
    return inspect_main(args)


def inspect_entry() -> None:
    raise SystemExit(main())
