#!/bin/bash
# Coding dogfood launch for a Slime GPU pod (e.g. RunPod A100 + slimerl/slime).
#
# Prerequisites (already done once on the pod):
#   - HF + torch_dist convert for the model you pass below
#   - pip install -e /root/daytona-training-gym-spec
#   - DAYTONA_API_KEY (+ optional DAYTONA_API_URL) exported
#
# Usage:
#   bash examples/coding_dogfood/run_on_slime_pod.sh
#   MODEL_SCRIPT=qwen2.5-3B.sh HF_CHECKPOINT=... REF_LOAD=... bash ...
#   # stress (more sandboxes + train steps):
#   BATCH_SIZE=4 N_SAMPLES=2 NUM_ROLLOUT=5 DAYTONA_MAX_CONCURRENCY=8 \
#     PROMPT_DATA=.../coding_pack.jsonl bash examples/coding_dogfood/run_on_slime_pod.sh

set -ex

REPO="${DAYTONA_GYM_REPO:-/root/daytona-training-gym-spec}"
SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/dogfood.jsonl}"
PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_one.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$((BATCH_SIZE * N_SAMPLES))}"
MAX_CONCURRENCY="${DAYTONA_MAX_CONCURRENCY:-$((BATCH_SIZE * N_SAMPLES))}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-9999}"
MODEL_SCRIPT="${MODEL_SCRIPT:-qwen2.5-0.5B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen2.5-0.5B-Instruct/}"
REF_LOAD="${REF_LOAD:-/root/Qwen2.5-0.5B-Instruct_torch_dist/}"
SGLANG_MEM="${SGLANG_MEM:-0.4}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.4}"
MAX_RESP_LEN="${MAX_RESP_LEN:-768}"

export REPO SLIME_ROOT MEGATRON_ROOT TELEMETRY_PATH PROMPT_DATA
export PYTHONUNBUFFERED=1
export DAYTONA_TELEMETRY_PATH="$TELEMETRY_PATH"
export DAYTONA_MAX_CONCURRENCY="$MAX_CONCURRENCY"
# Force coding seed + bootstrap into Ray workers via runtime_env (not just args).
export DAYTONA_SEED_CODING="${DAYTONA_SEED_CODING:-1}"
export DAYTONA_BOOTSTRAP_RUN_TESTS="${DAYTONA_BOOTSTRAP_RUN_TESTS:-1}"
export DAYTONA_BOOTSTRAP_RUN_TESTS_CMD="${DAYTONA_BOOTSTRAP_RUN_TESTS_CMD:-python test_broken.py}"
export DAYTONA_REQUIRE_PASSING_TESTS="${DAYTONA_REQUIRE_PASSING_TESTS:-1}"
mkdir -p "$(dirname "$TELEMETRY_PATH")"
: "${DAYTONA_API_KEY:?set DAYTONA_API_KEY}"

# Keep the API key out of Ray --runtime-env-json (ray job list dumps it).
# Workers resolve via DAYTONA_API_KEY_FILE → resolve_daytona_api_key().
KEY_FILE="${DAYTONA_API_KEY_FILE:-/tmp/daytona_gym_api_key}"
umask 077
printf '%s' "$DAYTONA_API_KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"
export DAYTONA_API_KEY_FILE="$KEY_FILE"

echo "=== Daytona dogfood scale ==="
echo "  prompts=$PROMPT_DATA"
echo "  batch=$BATCH_SIZE n_samples=$N_SAMPLES global_batch=$GLOBAL_BATCH"
echo "  num_rollout=$NUM_ROLLOUT steps_per_rollout=$NUM_STEPS_PER_ROLLOUT"
echo "  daytona_max_concurrency=$MAX_CONCURRENCY"
echo "  model=$MODEL_SCRIPT hf=$HF_CHECKPOINT"

echo "=== Daytona preflight (fail fast before Slime boot) ==="
python -m daytona_gym.preflight --timeout-seconds 90

# clean leftover ray/sglang
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 2

cd "$SLIME_ROOT"
source "${SLIME_ROOT}/scripts/models/${MODEL_SCRIPT}"

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$REF_LOAD"
   --save /tmp/slime_coding_dogfood_save/
   --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt
   --label-key label
   # Plain prompt: chat-template + naive concat after <|im_end|> corrupts multi-turn context.
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$BATCH_SIZE"
   --n-samples-per-prompt "$N_SAMPLES"
   --num-steps-per-rollout "$NUM_STEPS_PER_ROLLOUT"
   --global-batch-size "$GLOBAL_BATCH"
   --rollout-max-response-len "$MAX_RESP_LEN"
   --rollout-temperature "$ROLLOUT_TEMP"
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "$SGLANG_MEM"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

# Build runtime env JSON — never put DAYTONA_API_KEY here (ray job list dumps it).
RUNTIME_ENV=$(python3 - <<PY
import json, os, sys
env_vars = {
    "PYTHONPATH": f"{os.environ['MEGATRON_ROOT']}:{os.environ['REPO']}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "DAYTONA_API_KEY_FILE": os.environ["DAYTONA_API_KEY_FILE"],
    "DAYTONA_API_URL": os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
    "DAYTONA_TELEMETRY_PATH": os.environ["TELEMETRY_PATH"],
    "DAYTONA_MAX_CONCURRENCY": os.environ.get("DAYTONA_MAX_CONCURRENCY", "1"),
    "DAYTONA_SEED_CODING": os.environ.get("DAYTONA_SEED_CODING", "1"),
    "DAYTONA_BOOTSTRAP_RUN_TESTS": os.environ.get("DAYTONA_BOOTSTRAP_RUN_TESTS", "1"),
    "DAYTONA_BOOTSTRAP_RUN_TESTS_CMD": os.environ.get(
        "DAYTONA_BOOTSTRAP_RUN_TESTS_CMD", "python test_broken.py"
    ),
    "DAYTONA_REQUIRE_PASSING_TESTS": os.environ.get(
        "DAYTONA_REQUIRE_PASSING_TESTS", "1"
    ),
}
if "DAYTONA_API_KEY" in env_vars:
    print("refusing to put DAYTONA_API_KEY in Ray runtime_env", file=sys.stderr)
    raise SystemExit(2)
print("Ray runtime_env env_vars keys:", sorted(env_vars.keys()), file=sys.stderr)
print(json.dumps({"env_vars": env_vars}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   --custom-generate-function-path daytona_gym.adapters.slime.generate_dogfood.generate \
   --custom-rm-path daytona_gym.adapters.slime.reward.reward

echo "Inspect traces:"
echo "  dg"
echo "  dg --list-rollouts"
echo "  dg -r rollout_0"
