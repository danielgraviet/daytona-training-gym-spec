"""Short CLI entrypoints.

Examples:
  dg
  dg ls
  dg stats
  dg inspect
  dg runs/other.jsonl -r rollout_0
  python -m daytona_gym
"""

from __future__ import annotations

import sys

from daytona_gym.telemetry.inspect import main as inspect_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print(
            "dg — Daytona Gym helpers\n\n"
            "  dg                 inspect first rollout (runs/dogfood.jsonl)\n"
            "  dg ls [path]       list rollouts (status / reward / wall)\n"
            "  dg stats [path]    aggregate rewards, status, wall, tools\n"
            "  dg -r rollout_0    inspect one rollout\n"
            "  dg inspect [path]  same as dg [path]\n"
            "  dg i [path]        alias for inspect\n"
            "  dg --raw           legacy dense format\n"
        )
        return 0
    if args and args[0] in {"inspect", "i"}:
        args = args[1:]
    elif args and args[0] in {"ls", "list"}:
        args = ["--list-rollouts", *args[1:]]
    elif args and args[0] in {"stats", "summary", "sum"}:
        args = ["--stats", *args[1:]]
    return inspect_main(args)


def inspect_entry() -> None:
    raise SystemExit(main())
