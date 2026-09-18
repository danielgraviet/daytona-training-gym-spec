#!/bin/bash
# Coding dogfood launch for a Slime GPU pod (e.g. RunPod A100 + slimerl/slime).
#
# Prerequisites (already done once on the pod):
#   - /root/Qwen2.5-0.5B-Instruct + _torch_dist convert
#   - pip install -e /root/daytona-training-gym-spec
#   - DAYTONA_API_KEY (+ optional DAYTONA_API_URL) exported
#
# Usage (from anywhere on the pod):
#   bash /root/daytona-training-gym-spec/examples/coding_dogfood/run_on_slime_pod.sh

set -ex

REPO="${DAYTONA_GYM_REPO:-/root/daytona-training-gym-spec}"
SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
TELEMETRY_PATH="${DAYTONA_TELEMETRY_PATH:-$REPO/runs/dogfood.jsonl}"
PROMPT_DATA="${PROMPT_DATA:-$REPO/examples/coding_dogfood/prompts/coding_one.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$BATCH_SIZE}"
MAX_CONCURRENCY="${DAYTONA_MAX_CONCURRENCY:-$BATCH_SIZE}"

export REPO SLIME_ROOT MEGATRON_ROOT TELEMETRY_PATH PROMPT_DATA
export PYTHONUNBUFFERED=1
export DAYTONA_TELEMETRY_PATH="$TELEMETRY_PATH"
export DAYTONA_MAX_CONCURRENCY="$MAX_CONCURRENCY"
mkdir -p "$(dirname "$TELEMETRY_PATH")"
: "${DAYTONA_API_KEY:?set DAYTONA_API_KEY}"

# clean leftover ray/sglang
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 2

cd "$SLIME_ROOT"
source "${SLIME_ROOT}/scripts/models/qwen2.5-0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/
   --ref-load /root/Qwen2.5-0.5B-Instruct_torch_dist/
   --save /tmp/slime_coding_dogfood_save/
   --save-interval 9999
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --num-rollout 1
   --rollout-batch-size "$BATCH_SIZE"
   --n-samples-per-prompt "$N_SAMPLES"
   --num-steps-per-rollout 1
   --global-batch-size "$GLOBAL_BATCH"
   --rollout-max-response-len 512
   --rollout-temperature 0.7
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
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

# Build runtime env JSON with required secrets (do not commit the expanded file).
RUNTIME_ENV=$(python3 - <<PY
import json, os
print(json.dumps({
  "env_vars": {
    "PYTHONPATH": f"{os.environ['MEGATRON_ROOT']}:{os.environ['REPO']}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "DAYTONA_API_KEY": os.environ["DAYTONA_API_KEY"],
    "DAYTONA_API_URL": os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
    "DAYTONA_TELEMETRY_PATH": os.environ["TELEMETRY_PATH"],
  }
}))
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
echo "  python -m daytona_gym.telemetry.inspect $TELEMETRY_PATH"
echo "  python -m daytona_gym.telemetry.inspect $TELEMETRY_PATH --rollout rollout_0"
