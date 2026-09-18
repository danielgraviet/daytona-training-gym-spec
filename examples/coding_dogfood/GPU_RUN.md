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
- [ ] `sandbox.seed` + tool spans (`tool.run_tests` / `tool.write_file`) on coding prompt
- [x] Sample has tokens, loss_mask, rollout_log_probs, reward (train tensorize succeeded)
- [x] At least one optimizer/train step runs
- [x] Inspector timeline readable without a dashboard

Notes (2026-09-18 RunPod A100): math prompt completed as final text (no tool turns).
Coding seed + `run_on_slime_pod.sh` is the next GPU check. See `FRICTION.md`.
