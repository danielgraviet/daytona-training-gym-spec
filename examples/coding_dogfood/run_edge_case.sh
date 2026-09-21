#!/bin/bash
# Customer-shaped Slime × Daytona edge-case recipes for an existing GPU pod.
#
# Run ONE recipe, inspect, note FRICTION, then the next.
# Do not chain all recipes in one paste — failures/OOM poison later runs.
#
# Usage:
#   bash examples/coding_dogfood/run_edge_case.sh              # list recipes
#   bash examples/coding_dogfood/run_edge_case.sh tool_stall
#   dg stats runs/edge_tool_stall.jsonl
#
# Expectation: some recipes *should* abort / OOM / rate-limit. Capture with:
#   dg stats <path>
#   dg ls <path>
# and append notes to FRICTION.md.

set -euo pipefail

REPO="${DAYTONA_GYM_REPO:-/root/daytona-training-gym-spec}"
RECIPE="${1:-}"
shift || true

# Shared 3B defaults (override freely).
export MODEL_SCRIPT="${MODEL_SCRIPT:-qwen2.5-3B.sh}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen2.5-3B-Instruct/}"
export REF_LOAD="${REF_LOAD:-/root/Qwen2.5-3B-Instruct_torch_dist/}"
export ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.2}"

_usage() {
  cat <<'EOF'
Run ONE recipe at a time (inspect + FRICTION note before the next).

Recipes (each writes its own telemetry file):

  tool_stall       Bootstrap sleeps 120s; tool timeout 5s → expect tool_timeout abort
  rollout_budget   Tight whole-rollout timeout (default 3s) during coding loop
  wrong_bootstrap  Bootstrap command missing → soft fail, model must recover
  hard_prompts     Multi-file seed + prompts that do not spoon-feed the fix
  concurrency_storm  Many sandboxes at once (rate limits / queueing)
  mem_pressure     High SGLang mem + longer responses (OOM / thrash risk) — run last

Order tip: failures first, mem_pressure last.

Examples:
  bash examples/coding_dogfood/run_edge_case.sh tool_stall
  dg stats runs/edge_tool_stall.jsonl
EOF
}

if [[ -z "$RECIPE" || "$RECIPE" == "-h" || "$RECIPE" == "--help" ]]; then
  _usage
  exit 0
fi

case "$RECIPE" in
  tool_stall)
    export DAYTONA_TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/edge_tool_stall.jsonl}"
    export DAYTONA_RUN_ID="edge_tool_stall"
    export DAYTONA_BOOTSTRAP_RUN_TESTS_CMD='python -c "import time; time.sleep(120)"'
    export DAYTONA_TOOL_TIMEOUT_SECONDS="${DAYTONA_TOOL_TIMEOUT_SECONDS:-5}"
    export DAYTONA_TIMEOUT_SECONDS="${DAYTONA_TIMEOUT_SECONDS:-45}"
    export BATCH_SIZE=1 N_SAMPLES=1 NUM_ROLLOUT=1
    export DAYTONA_MAX_CONCURRENCY=1
    export PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_one.jsonl}"
    # Default: fail the job loudly so we see customer-shaped abort. Override with ALLOW_ABORTED=1.
    export DAYTONA_ALLOW_ABORTED="${DAYTONA_ALLOW_ABORTED:-0}"
    ;;
  rollout_budget)
    export DAYTONA_TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/edge_rollout_budget.jsonl}"
    export DAYTONA_RUN_ID="edge_rollout_budget"
    # Happy-path coding on 3B/H100 is ~6s end-to-end; 6s never trips. Stay under
    # provision+seed+bootstrap (~4s) so the whole-rollout budget fires mid-loop.
    export DAYTONA_TIMEOUT_SECONDS="${DAYTONA_TIMEOUT_SECONDS:-3}"
    export DAYTONA_TOOL_TIMEOUT_SECONDS="${DAYTONA_TOOL_TIMEOUT_SECONDS:-30}"
    export BATCH_SIZE=1 N_SAMPLES=1 NUM_ROLLOUT=1
    export DAYTONA_MAX_CONCURRENCY=1
    export DAYTONA_MAX_TURNS=8
    export PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_one.jsonl}"
    export DAYTONA_ALLOW_ABORTED="${DAYTONA_ALLOW_ABORTED:-0}"
    ;;
  concurrency_storm)
    export DAYTONA_TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/edge_concurrency.jsonl}"
    export DAYTONA_RUN_ID="edge_concurrency"
    export BATCH_SIZE="${BATCH_SIZE:-8}"
    export N_SAMPLES="${N_SAMPLES:-2}"
    export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
    export DAYTONA_MAX_CONCURRENCY="${DAYTONA_MAX_CONCURRENCY:-16}"
    export PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_pack.jsonl}"
    export SGLANG_MEM="${SGLANG_MEM:-0.5}"
    # Keep going if a subset abort (rate limit / timeout); still raise on hard provision fail.
    export DAYTONA_ALLOW_ABORTED="${DAYTONA_ALLOW_ABORTED:-1}"
    export DAYTONA_TIMEOUT_SECONDS="${DAYTONA_TIMEOUT_SECONDS:-180}"
    export DAYTONA_TOOL_TIMEOUT_SECONDS="${DAYTONA_TOOL_TIMEOUT_SECONDS:-60}"
    ;;
  mem_pressure)
    export DAYTONA_TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/edge_mem_pressure.jsonl}"
    export DAYTONA_RUN_ID="edge_mem_pressure"
    export BATCH_SIZE="${BATCH_SIZE:-4}"
    export N_SAMPLES="${N_SAMPLES:-2}"
    export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
    export DAYTONA_MAX_CONCURRENCY="${DAYTONA_MAX_CONCURRENCY:-8}"
    export SGLANG_MEM="${SGLANG_MEM:-0.85}"
    export MAX_RESP_LEN="${MAX_RESP_LEN:-2048}"
    export PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_pack.jsonl}"
    export DAYTONA_ALLOW_ABORTED="${DAYTONA_ALLOW_ABORTED:-1}"
    ;;
  hard_prompts)
    export DAYTONA_TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/edge_hard_prompts.jsonl}"
    export DAYTONA_RUN_ID="edge_hard_prompts"
    export DAYTONA_SEED_PROFILE=multifile
    export PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_hard.jsonl}"
    export BATCH_SIZE="${BATCH_SIZE:-2}"
    export N_SAMPLES="${N_SAMPLES:-2}"
    export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
    export DAYTONA_MAX_CONCURRENCY="${DAYTONA_MAX_CONCURRENCY:-4}"
    export DAYTONA_MAX_TURNS="${DAYTONA_MAX_TURNS:-8}"
    export DAYTONA_MAX_TOOLS_PER_TURN="${DAYTONA_MAX_TOOLS_PER_TURN:-3}"
    export DAYTONA_ALLOW_ABORTED="${DAYTONA_ALLOW_ABORTED:-1}"
    ;;
  wrong_bootstrap)
    export DAYTONA_TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/edge_wrong_bootstrap.jsonl}"
    export DAYTONA_RUN_ID="edge_wrong_bootstrap"
    export DAYTONA_BOOTSTRAP_RUN_TESTS_CMD='python this_file_does_not_exist.py'
    export PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_one.jsonl}"
    export BATCH_SIZE=1 N_SAMPLES=1 NUM_ROLLOUT=1
    export DAYTONA_MAX_CONCURRENCY=1
    export DAYTONA_ALLOW_ABORTED="${DAYTONA_ALLOW_ABORTED:-0}"
    ;;
  *)
    echo "unknown recipe: $RECIPE" >&2
    _usage >&2
    exit 2
    ;;
esac

rm -f "$DAYTONA_TELEMETRY_PATH"
echo "=== edge recipe: $RECIPE ==="
echo "  telemetry=$DAYTONA_TELEMETRY_PATH"
echo "  extra args: $*"

bash "$REPO/examples/coding_dogfood/run_on_slime_pod.sh" "$@"
echo
echo "=== inspect ==="
echo "  dg $DAYTONA_TELEMETRY_PATH"
echo "  dg ls $DAYTONA_TELEMETRY_PATH"
echo "  dg stats $DAYTONA_TELEMETRY_PATH"
