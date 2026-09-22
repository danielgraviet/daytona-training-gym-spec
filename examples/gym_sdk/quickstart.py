#!/usr/bin/env python3
"""Minimal Modal-shaped Gym SDK example (dry-run by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

from daytona_gym import (
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
)

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        model=Qwen25_3B(),
        dataset=PromptJsonlDataset(
            REPO / "examples/coding_dogfood/prompts/coding_one.jsonl"
        ),
        recipe=Qwen25_3B_Recipe(),
        repo=REPO,
    )
    run = config.launch(dry_run=not args.launch)
    print(run.training_run_id)
    print(run.inspect_hint)
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
