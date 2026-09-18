# Coding dogfood example
#
# Goal: cheapest path to one real Slime train step with Daytona sandboxes
# and functional JSONL traces. Launch Slime yourself on your GPU; Daytona
# only plugs custom generate + reward.

## What this proves

1. Slime scheduler calls `daytona_gym.adapters.slime.generate`
2. That hook calls **your** SGLang router for tokens
3. Tool turns run in a Daytona sandbox
4. Sample returns with tokens / loss_mask / rollout_log_probs / reward
5. Megatron takes at least one train step
6. Traces land in `runs/dogfood.jsonl` and print via the inspector

## Prerequisites

- Working Slime install that can already train a small model (e.g. Qwen3-4B)
- `DAYTONA_API_KEY` (and optional `DAYTONA_API_URL`) in the environment
- `pip install -e .` from this repo (or `uv sync`)
- A coding snapshot/image with a tiny failing test the agent can fix

Suggested tiny task layout inside the sandbox image/snapshot:

```text
/workspace/
  broken.py      # def add(a, b): return a - b
  test_broken.py # assert add(1, 2) == 3
```

Prompt example:

```text
Fix the failing tests in /workspace. Use run_tests with pytest.
When tests pass, respond with {"type":"final","content":"fixed"}.
```

Agent tool protocol (model must emit JSON lines):

```json
{"type":"tool","name":"run_tests","arguments":{"command":"pytest"}}
{"type":"tool","name":"write_file","arguments":{"path":"broken.py","content":"..."}}
{"type":"final","content":"fixed"}
```

## Slime launch sketch

Wire Daytona into an existing Slime script. Keep your normal Megatron/SGLang
flags; add only the hooks and Daytona args:

```bash
export DAYTONA_API_KEY=...
export DAYTONA_TELEMETRY_PATH="${PWD}/runs/dogfood.jsonl"
mkdir -p runs

python train.py \
  ...your normal slime / megatron / sglang args... \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --custom-generate-function-path daytona_gym.adapters.slime.generate \
  --custom-rm-path daytona_gym.adapters.slime.reward.reward
```

Pass Daytona settings on the Slime `args` object (however your train script
exposes custom attributes), for example:

| arg | purpose |
|---|---|
| `daytona_image` or `daytona_snapshot` | sandbox base |
| `daytona_max_turns` | agent loop bound (default 8) |
| `daytona_max_concurrency` | sandbox cap |
| `daytona_telemetry_path` | JSONL export path |
| `daytona_run_id` | stable run id |
| `daytona_project_id` | correlation |
| `daytona_return_logprob` | keep `True` for training |

SGLang router fields (`sglang_router_ip`, `sglang_router_port`) are set by
Slime when it starts the router. Do not set `daytona_generator` in production;
the adapter defaults to `SGLangRouterGenerator`.

Minimal glue if your train entrypoint uses a namespace/argparse object:

```python
args.daytona_snapshot = "your-coding-snapshot"
args.daytona_telemetry_path = "runs/dogfood.jsonl"
args.daytona_run_id = "dogfood_1"
args.daytona_project_id = "coding-rl"
args.daytona_max_turns = 6
args.daytona_max_concurrency = 2
args.daytona_return_logprob = True
```

See `inject_daytona_args.py` for a copy-paste helper.

## After the run

```bash
uv run python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl
uv run python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0
```

You should see spans roughly like:

```text
rollout
sandbox.provision
inference.generate
tool.run_tests / tool.write_file / ...
sandbox.finalize
```

Reward is inferred from the last `run_tests` tool result during generate and
exposed to Slime via `--custom-rm-path daytona_gym.adapters.slime.reward.reward`.

## Cost knobs

- smallest model you already run
- `--num-rollout 1` / batch size 1
- `daytona_max_concurrency=1` or `2`
- short `daytona_timeout_seconds`
- telemetry path on local disk (ring buffer already caps RAM)

## Friction log

Capture anything painful during dogfood in `FRICTION.md` in this folder
(setup, missing args, unclear errors). That becomes the partner backlog.
