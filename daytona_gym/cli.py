"""Short CLI entrypoints.

Examples:
  dg
  dg ls
  dg stats
  dg dash
  dg open
  dg run main.py
  dg ingest
  dg run list|status|logs|wait|stop
  dg skills install
  dg inspect
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
            "  dg dash            live dashboard on http://127.0.0.1:3000 (SPA)\n"
            "                     auto-pulls RunPod runs/ from .env (PtyShell)\n"
            "  dg open            alias for dg dash\n"
            "  dg dash --port N   optional port (default 3000)\n"
            "  dg dash --share    optional public Cloudflare tunnel\n"
            "  dg dash --remote USER@HOST   escape hatch (non-RunPod SSH)\n"
            "  dg ingest          durable run history + dashboard (workers ship here)\n"
            "  dg run main.py     run a Modal-shaped train script (or uv run)\n"
            "  dg run list|status|logs|wait|stop\n"
            "  dg skills install  install agent skill bundle\n"
            "  dg -r rollout_0    inspect one rollout\n"
            "  dg inspect [path]  same as dg [path]\n"
            "  dg i [path]        alias for inspect\n"
            "  dg --raw           legacy dense format\n\n"
            "Gym SDK (Python):\n"
            "  from daytona_gym import TrainConfig, HuggingFaceDataset, runpod_worker\n"
            "  run = TrainConfig(...).launch(worker=runpod_worker(), detach=True)\n"
        )
        return 0
    if args and args[0] in {"dash", "open", "dashboard"}:
        from daytona_gym.telemetry.dashboard import main as dash_main

        return dash_main(args[1:])
    if args and args[0] == "ingest":
        from daytona_gym.ingest.server import main as ingest_main

        return ingest_main(args[1:])
    if args and args[0] == "run":
        from daytona_gym.gym.run_cli import main as run_main

        return run_main(args[1:])
    if args and args[0] == "skills":
        from daytona_gym.skills import main as skills_main

        return skills_main(args[1:])
    if args and args[0] in {"inspect", "i"}:
        args = args[1:]
    elif args and args[0] in {"ls", "list"}:
        args = ["--list-rollouts", *args[1:]]
    elif args and args[0] in {"stats", "summary", "sum"}:
        args = ["--stats", *args[1:]]
    return inspect_main(args)


def inspect_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
