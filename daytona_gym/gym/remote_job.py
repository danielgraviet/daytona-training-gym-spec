"""Run on a GPU worker: ``python -m daytona_gym.gym.remote_job /tmp/job.json``.

Prints machine-parseable markers for the laptop-side ``SshWorker``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from daytona_gym.gym.compute import LocalSlimeCompute
from daytona_gym.gym.config import TrainConfig
from daytona_gym.gym.dataset import PromptJsonlDataset
from daytona_gym.gym.models import SoftSlimeModel
from daytona_gym.gym.recipe import CodingRecipe
from daytona_gym.runtime.errors import DaytonaError


def load_config(payload: dict) -> TrainConfig:
    dataset = PromptJsonlDataset(
        path=payload["dataset"]["path"],
        input_key=payload["dataset"].get("input_key", "prompt"),
        label_key=payload["dataset"].get("label_key", "label"),
    )
    recipe = CodingRecipe(**payload["recipe"])
    model = None
    compute = None
    if payload.get("model"):
        model = SoftSlimeModel(**payload["model"])
    if payload.get("compute"):
        compute = LocalSlimeCompute(**payload["compute"], repo=payload.get("repo"))
    return TrainConfig(
        dataset=dataset,
        recipe=recipe,
        model=model,
        compute=compute,
        run_name=payload.get("run_name"),
        telemetry_path=payload.get("telemetry_path"),
        repo=payload.get("repo"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_json", type=Path, help="Path to job payload JSON")
    args = parser.parse_args(argv)

    payload = json.loads(args.job_json.read_text(encoding="utf-8"))
    config = load_config(payload)
    try:
        run = config._launch_local(
            dry_run=False,
            skip_preflight=bool(payload.get("skip_preflight", False)),
            preflight_timeout_seconds=float(
                payload.get("preflight_timeout_seconds", 90)
            ),
            open=bool(payload.get("open", True)),
            open_browser=bool(payload.get("open_browser", False)),
        )
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        print("__DG_RETURNCODE__=2", flush=True)
        return 2

    print(f"__DG_RUN_ID__={run.training_run_id}", flush=True)
    print(f"__DG_TELEMETRY__={run.telemetry_path}", flush=True)
    if run.dashboard_url:
        print(f"__DG_DASHBOARD__={run.dashboard_url}", flush=True)
    print(f"__DG_RETURNCODE__={int(run.returncode or 0)}", flush=True)

    if run.dashboard_url:
        run.wait_dashboard()
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
