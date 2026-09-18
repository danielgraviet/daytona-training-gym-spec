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
- [ ] Tool spans (`tool.run_tests` / `tool.write_file`) — blocked on 0.5B (no JSON tools; `reward=None`)
- [x] Sample has tokens, loss_mask, rollout_log_probs (train tensorize succeeded); reward only when tests run
- [x] At least one optimizer/train step runs
- [x] Inspector timeline readable without a dashboard

Notes (2026-09-18 RunPod A100): coding seed works; 0.5B skips tools; concurrency=2 cleanup passed (`raysubmit_Nn9JEmvhrNjY88Vt`). See `FRICTION.md`.
