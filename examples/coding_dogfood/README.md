# Coding dogfood example
#
# Goal: one real Slime train step with Daytona sandboxes, tool turns, and
# functional JSONL traces. Prefer `run_on_slime_pod.sh` on a GPU box.

## What this proves

1. Slime calls `daytona_gym.adapters.slime.generate_dogfood.generate`
2. Sandbox is seeded with a broken `add()` task
3. Model uses tools (`run_tests` / `write_file`) or at least hits seed + generate
4. Sample returns tokens / loss_mask / rollout_log_probs / reward
5. Megatron takes at least one train step
6. Traces land in JSONL and print via the inspector (`sandbox.seed`, tools, …)

## On a Slime GPU pod (RunPod)

```bash
cd /root/daytona-training-gym-spec
git pull
pip install -e .

export DAYTONA_API_KEY=...
export DAYTONA_API_URL=https://app.daytona.io/api   # if needed

bash examples/coding_dogfood/run_on_slime_pod.sh

python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0
```

Expect spans like:

```text
rollout
sandbox.provision
sandbox.seed
inference.generate
tool.run_tests / tool.write_file   # if the model follows the JSON protocol
sandbox.finalize
```

Note: Qwen2.5-0.5B often ignores tool JSON. A successful train step + `sandbox.seed`
still validates the path; tool turns are more reliable on ≥1.5B/4B or with a
stronger instruct checkpoint.

## Seeded task

| file | content |
|---|---|
| `broken.py` | `add` returns `a - b` (wrong) |
| `test_broken.py` | asserts `add(1, 2) == 3` via plain `python` (no pytest) |

Override with `args.daytona_seed_files = {...}` if needed.

## Pass criteria

See `GPU_RUN.md`. Capture pain in `FRICTION.md`.
