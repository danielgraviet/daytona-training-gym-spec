#!/usr/bin/env python3
"""RunPod dogfood for the Daytona Gym SDK (slimerl/slime:latest).

Prerequisites: a fresh pod from image ``slimerl/slime:latest`` with a GPU,
and a Daytona API key.

Exact steps on the pod:

  # 1) clone + install
  cd /root
  git clone https://github.com/danielgraviet/daytona-training-gym-spec.git
  cd /root/daytona-training-gym-spec
  pip install -e .

  # 2) Daytona credentials (key stays in a file; never in Ray runtime_env)
  export DAYTONA_API_KEY='...'
  # export DAYTONA_API_URL=https://app.daytona.io/api   # only if non-default

  # 3) download + convert model (3B — proven coding tools; skip dirs if present)
  hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct
  cd /root/slime
  source scripts/models/qwen2.5-3B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \\
    ${MODEL_ARGS[@]} \\
    --hf-checkpoint /root/Qwen2.5-3B-Instruct \\
    --save /root/Qwen2.5-3B-Instruct_torch_dist

  #    smaller first smoke (optional instead of step 3):
  #      hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct
  #      cd /root/slime && source scripts/models/qwen2.5-0.5B.sh
  #      PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \\
  #        ${MODEL_ARGS[@]} \\
  #        --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \\
  #        --save /root/Qwen2.5-0.5B-Instruct_torch_dist
  #      export MODEL_SCRIPT=qwen2.5-0.5B.sh
  #      export HF_CHECKPOINT=/root/Qwen2.5-0.5B-Instruct/
  #      export REF_LOAD=/root/Qwen2.5-0.5B-Instruct_torch_dist/

  # 4) dry-run (no GPU train — prints Ray/Slime command)
  cd /root/daytona-training-gym-spec
  python examples/gym_sdk/runpod_dogfood.py

  # 5) real one-step coding dogfood
  python examples/gym_sdk/runpod_dogfood.py --launch

  # 6) inspect traces (use the run_id printed in step 5)
  dg stats runs/<run_id>.jsonl
  dg ls runs/<run_id>.jsonl

Expected layout after setup:

  /root/slime
  /root/Megatron-LM
  /root/daytona-training-gym-spec
  /root/Qwen2.5-3B-Instruct/
  /root/Qwen2.5-3B-Instruct_torch_dist/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from daytona_gym import (
    CodingRecipe,
    LocalSlimeCompute,
    PromptJsonlDataset,
    TrainConfig,
    TrainingRun,
)
from daytona_gym.runtime.errors import DaytonaError

# --- paths: override with env vars if your layout differs ---
REPO = Path(os.environ.get("DAYTONA_GYM_REPO", Path(__file__).resolve().parents[2]))
SLIME_ROOT = os.environ.get("SLIME_ROOT", "/root/slime")
MEGATRON_ROOT = os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM")
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "/root/Qwen2.5-3B-Instruct/")
REF_LOAD = os.environ.get("REF_LOAD", "/root/Qwen2.5-3B-Instruct_torch_dist/")
MODEL_SCRIPT = os.environ.get("MODEL_SCRIPT", "qwen2.5-3B.sh")
PROMPT_DATA = os.environ.get(
    "PROMPT_DATA",
    str(REPO / "examples" / "coding_dogfood" / "prompts" / "coding_one.jsonl"),
)


def build_config() -> TrainConfig:
    return TrainConfig(
        compute=LocalSlimeCompute(
            slime_root=SLIME_ROOT,
            megatron_root=MEGATRON_ROOT,
            hf_checkpoint=HF_CHECKPOINT,
            ref_load=REF_LOAD,
            model_script=MODEL_SCRIPT,
            repo=REPO,
            sglang_mem_fraction=float(os.environ.get("SGLANG_MEM", "0.4")),
            save_dir=os.environ.get("SLIME_SAVE_DIR", "/tmp/slime_coding_dogfood_save"),
        ),
        dataset=PromptJsonlDataset(
            path=PROMPT_DATA,
            input_key="prompt",
            label_key="label",
        ),
        recipe=CodingRecipe(
            batch_size=int(os.environ.get("BATCH_SIZE", "1")),
            n_samples=int(os.environ.get("N_SAMPLES", "1")),
            num_rollout=int(os.environ.get("NUM_ROLLOUT", "1")),
            max_concurrency=int(os.environ.get("DAYTONA_MAX_CONCURRENCY", "1")),
            max_response_len=int(os.environ.get("MAX_RESP_LEN", "768")),
            temperature=float(os.environ.get("ROLLOUT_TEMP", "0.4")),
            seed_coding=True,
            seed_profile=os.environ.get("DAYTONA_SEED_PROFILE", "basic"),
            bootstrap_run_tests=True,
            bootstrap_run_tests_cmd=os.environ.get(
                "DAYTONA_BOOTSTRAP_RUN_TESTS_CMD", "python test_broken.py"
            ),
            require_passing_tests=True,
            allow_aborted=os.environ.get("DAYTONA_ALLOW_ABORTED", "0") == "1",
            generate_path="daytona_gym.adapters.slime.generate_dogfood.generate",
            rm_path="daytona_gym.adapters.slime.reward.reward",
        ),
        run_name=os.environ.get("DAYTONA_RUN_ID"),
        telemetry_path=os.environ.get("DAYTONA_TELEMETRY_PATH"),
    )


def print_run(run: TrainingRun) -> None:
    print(f"run_id={run.run_id}")
    print(f"telemetry={run.telemetry_path}")
    print(f"dry_run={run.dry_run}")
    print(f"returncode={run.returncode}")
    print(f"inspect: {run.inspect_hint}")
    print(f"dg ls {run.telemetry_path}")
    env_vars = run.runtime_env.get("env_vars", {})
    print("runtime_env keys:", sorted(env_vars.keys()))
    if "DAYTONA_API_KEY" in env_vars:
        print("BUG: DAYTONA_API_KEY leaked into Ray runtime_env", file=sys.stderr)
    print("command:")
    print(" ", " ".join(run.command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Execute on this GPU host (default is dry-run only)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Daytona sandbox preflight (debug only)",
    )
    args = parser.parse_args(argv)

    config = build_config()
    try:
        if args.launch:
            run = config.launch(
                dry_run=False,
                skip_preflight=args.skip_preflight,
            )
        else:
            run = config.launch(dry_run=True)
    except DaytonaError as exc:
        print(f"daytona error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2

    print_run(run)
    if run.dry_run:
        print("\nRe-run with --launch when ready on the GPU box.")
        return 0
    return int(run.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
