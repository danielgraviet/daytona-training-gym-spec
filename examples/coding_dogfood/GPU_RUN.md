# GPU dogfood run checklist

Software path is validated on CPU via:

```bash
uv run python -m pytest -q
uv run python scripts/validate_dogfood_path.py
```

## On the GPU box (coding tool-loop)

```bash
cd /root/daytona-training-gym-spec && git pull && pip install -e .
export DAYTONA_API_KEY=...
# export DAYTONA_API_URL=https://app.daytona.io/api
rm -f runs/dogfood.jsonl
bash examples/coding_dogfood/run_on_slime_pod.sh
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0
# or short:
dg
```

Expect a log line like `[daytona-gym] generate seed=[...] bootstrap='python test_broken.py'`.

After submit, `ray job list` should show `DAYTONA_API_KEY_FILE` (path only), **not** `DAYTONA_API_KEY`.

## Pass criteria

- [x] Sandbox provisioned and cleaned up
- [x] SGLang ↔ Daytona loop visible in JSONL (`sandbox.provision` / `inference.generate` / `sandbox.finalize`)
- [x] `sandbox.seed` on coding prompt (`run_on_slime_pod.sh`)
- [x] Bootstrap `tool.run_tests` (forced after seed; model-emitted tools still optional)
- [x] Model-emitted `tool.write_file` / second `run_tests` on Qwen2.5-3B (parser fix)
- [x] Coding dogfood `reward=1.0` (correct `a + b` fix) — `raysubmit_Ah8AjLWwnxfDK7Qe`
- [x] Sample has tokens, loss_mask, rollout_log_probs (train tensorize succeeded); reward from bootstrap tests
- [x] At least one optimizer/train step runs
- [x] Inspector timeline readable without a dashboard

## Verified timeline (2026-09-18, job `raysubmit_qMmVkw1ZeGapWCCi`)

```text
sandbox.provision → sandbox.seed → tool.run_tests → inference.generate → sandbox.finalize
tools run_tests=1
```

Notes: coding seed + concurrency=2 + preflight OK on RunPod A100; 0.5B/1.5B skip JSON tools — bootstrap supplies `tool.run_tests`. Bootstrap must travel via Ray `runtime_env` env vars (`DAYTONA_BOOTSTRAP_RUN_TESTS*` / `DAYTONA_SEED_CODING`), not only `args` setattr. See `FRICTION.md`.

## Stress / scale (H100)

More sandboxes ≈ `BATCH_SIZE * N_SAMPLES` (capped by `DAYTONA_MAX_CONCURRENCY`).
More train updates ≈ `NUM_ROLLOUT`.

```bash
cd /root/daytona-training-gym-spec && git pull && pip install -e .
rm -f runs/dogfood.jsonl

MODEL_SCRIPT=qwen2.5-3B.sh \
  HF_CHECKPOINT=/root/Qwen2.5-3B-Instruct/ \
  REF_LOAD=/root/Qwen2.5-3B-Instruct_torch_dist/ \
  PROMPT_DATA=/root/daytona-training-gym-spec/examples/coding_dogfood/prompts/coding_pack.jsonl \
  BATCH_SIZE=4 \
  N_SAMPLES=2 \
  NUM_ROLLOUT=5 \
  DAYTONA_MAX_CONCURRENCY=8 \
  SGLANG_MEM=0.5 \
  ROLLOUT_TEMP=0.2 \
  bash examples/coding_dogfood/run_on_slime_pod.sh

dg --list-rollouts
dg -r rollout_0
```

If Daytona rate-limits or the GPU OOMs, drop to `BATCH_SIZE=2` / `DAYTONA_MAX_CONCURRENCY=4`.
