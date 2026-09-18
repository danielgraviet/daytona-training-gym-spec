# Dogfood friction log

## Environment

- Date: 2026-09-18
- GPU / machine: Cursor agent host (no NVIDIA detected); live Megatron step deferred to Daniel's GPU box
- Slime commit / version: not installed on agent host (`NO_SLIME`)
- Model: TBD on GPU box
- Daytona image/snapshot: TBD on GPU box

## What worked (CPU / software path)

- `uv run python -m pytest -q` — 76+ tests green including SGLang fake-router, sample logprobs, inspect CLI
- `uv run python scripts/validate_dogfood_path.py` — default `SGLangRouterGenerator` wiring, fake sandbox multi-turn tools, JSONL export, inspector timeline
- Reward path reads `metadata["daytona"]["reward"]` via `daytona_gym.adapters.slime.reward.reward`

## What hurt

- Agent host cannot run the real Slime+Megatron step (no Slime package, no GPU)
- Live GPU train step must be executed on Daniel's machine per `GPU_RUN.md`

## Missing product pieces (for external partner)

- Still no BYO GPU worker agent / dashboard (intentionally deferred)
- Need a real coding snapshot ID documented once dogfood image is chosen
- Launch still requires manual Slime args glue (`inject_daytona_args.py`)

## Errors / stack traces worth keeping

```text
(none on CPU dry-run)
```
