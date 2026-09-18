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
bash examples/coding_dogfood/run_on_slime_pod.sh
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl
```

## Pass criteria

- [x] Sandbox provisioned and cleaned up
- [x] SGLang ↔ Daytona loop visible in JSONL (`sandbox.provision` / `inference.generate` / `sandbox.finalize`)
- [x] `sandbox.seed` on coding prompt (`run_on_slime_pod.sh`)
- [x] Bootstrap `tool.run_tests` (forced after seed; model-emitted tools still optional)
- [ ] Model-emitted `tool.write_file` / second `run_tests` (needs stronger model or more turns)
- [x] Sample has tokens, loss_mask, rollout_log_probs (train tensorize succeeded); reward from bootstrap tests
- [x] At least one optimizer/train step runs
- [x] Inspector timeline readable without a dashboard

Notes (2026-09-18 RunPod A100): coding seed + concurrency=2 + preflight OK; 0.5B/1.5B skip JSON tools — use bootstrap. See `FRICTION.md`.
