# GPU dogfood run checklist

Software path is validated on CPU via:

```bash
uv run python -m pytest -q
uv run python scripts/validate_dogfood_path.py
```

This machine may not have Slime or NVIDIA GPUs. Complete the items
below on **your** GPU box.

## On the GPU box

1. Install this package into the Slime env: `pip install -e /path/to/daytona-training-gym-spec`
2. Confirm Slime already trains a small model without Daytona
3. Set `DAYTONA_API_KEY` (+ optional `DAYTONA_API_URL`)
4. Point a coding snapshot/image with a tiny failing test
5. Inject Daytona args (`examples/coding_dogfood/inject_daytona_args.py`)
6. Launch Slime with:
   - `--custom-generate-function-path daytona_gym.adapters.slime.generate`
   - `--custom-rm-path daytona_gym.adapters.slime.reward.reward`
   - `daytona_telemetry_path=runs/dogfood.jsonl`
   - 1 rollout / batch size 1 / low concurrency
7. Confirm one Megatron train step completes (loss logged; no crash)
8. Inspect traces:

```bash
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0
```

9. Fill `FRICTION.md` with anything painful

## Pass criteria

- [ ] Sandbox provisioned and cleaned up
- [ ] Multi-turn SGLang ↔ tool loop visible in JSONL
- [ ] Sample has tokens, loss_mask, rollout_log_probs, reward
- [ ] At least one optimizer/train step runs
- [ ] Inspector timeline readable without a dashboard
