# Analytics-only — dashboard without the Gym launcher

Use Daytona Gym's **adapter + ingest** with your existing Slime command.
No `TrainConfig`, no BYO worker helper, no Ray job wrapper — just the
custom-generate hook and a telemetry shipper.

**Done when:** a Slime run you already launch (laptop, Modal, RunPod, Vast, …)
shows up under `dg ingest` with the full "where time went" view.

---

## 1. One-time: durable dashboard host

```bash
# Long-lived Daytona sandbox (or any HTTPS host running `dg ingest`)
dg ingest deploy --daytona
# prints DAYTONA_GYM_INGEST_URL + DAYTONA_GYM_INGEST_TOKEN — keep them
```

## 2. On the GPU box (env)

```bash
pip install -e /path/to/daytona-training-gym-spec   # or: pip install daytona-gym

export DAYTONA_API_KEY=...                          # sandboxes
export DAYTONA_GYM_INGEST_URL=https://…             # from step 1
export DAYTONA_GYM_INGEST_TOKEN=…                   # ≥16 chars
export DAYTONA_TELEMETRY_PATH=runs/analytics_only.jsonl
export DAYTONA_RUN_ID=run_$(date +%Y%m%d_%H%M%S)    # stable id for the dashboard
# optional: DAYTONA_SEED_CODING=1 for the toy add(a,b) seed
```

## 3. Point Slime at the adapter

Add (or keep) these flags on **your** `train.py` invocation:

```bash
--custom-generate-function-path daytona_gym.adapters.slime.generate.generate \
--custom-rm-path daytona_gym.adapters.slime.reward.reward
```

Everything else (model, Megatron, dataset, GRPO knobs) stays yours.

## 4. Ship telemetry while training

In a **second terminal** on the same machine (JSONL is written locally first):

```bash
dg ship -f "$DAYTONA_TELEMETRY_PATH"
# or: python -m daytona_gym.ingest.shipper -f runs/analytics_only.jsonl
```

Open the printed URL (or `$DAYTONA_GYM_INGEST_URL/run/$DAYTONA_RUN_ID`).

After the job finishes, a one-shot backfill is enough:

```bash
dg ship runs/analytics_only.jsonl
# same as: dg ingest push runs/analytics_only.jsonl
```

## 5. What you get

- Per-rollout sandbox / tool / inference spans
- Step-level “where time went” + optional `$` if you later set a GPU rate
- Runs that outlive the GPU box (`worker_lost` if the shipper dies mid-run)

## Minimal copy-paste

```bash
export DAYTONA_API_KEY=…
export DAYTONA_GYM_INGEST_URL=…
export DAYTONA_GYM_INGEST_TOKEN=…
export DAYTONA_TELEMETRY_PATH=runs/analytics_only.jsonl
export DAYTONA_RUN_ID=run_analytics_demo

dg ship -f "$DAYTONA_TELEMETRY_PATH" &
SHIP_PID=$!

python train.py \
  …your slime args… \
  --custom-generate-function-path daytona_gym.adapters.slime.generate.generate \
  --custom-rm-path daytona_gym.adapters.slime.reward.reward

kill $SHIP_PID
dg ship "$DAYTONA_TELEMETRY_PATH"   # final drain
echo "dashboard: $DAYTONA_GYM_INGEST_URL/run/$DAYTONA_RUN_ID"
```

Prefer the full Gym SDK (`TrainConfig.launch`) when you also want dataset shipping,
recipe presets, and detach/dashboard lifecycle — this page is for teams that
already have a Slime launch and only want the analytics layer.
