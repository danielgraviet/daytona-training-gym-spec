# Daytona Training Gym — MVP Spec

## Goal

Build a portable reinforcement-learning post-training **gym** where users can:

1. bring their own GPUs,
2. use an existing RL framework (Slime first; Harbor later as a backend),
3. run agentic rollouts inside Daytona sandboxes,
4. automatically collect rollout + training + inference telemetry,
5. inspect one unified dashboard for bottlenecks and failures.

Daytona should **not** become a new RL training framework.

The product boundary is:

> Existing RL framework owns optimization. Existing inference engine owns token generation. Daytona owns stateful rollout environments, rollout execution infrastructure, and cross-layer observability.

**North star UX** matches [Modal Training Gym](https://gym.modal.dev/) (`TrainConfig.launch()` + recipes + run handle + dashboard), with the explicit diffs **BYO GPUs** and **Daytona sandboxes**. See [`COMPETITIVE_MODAL_GYM.md`](COMPETITIVE_MODAL_GYM.md).

## First integration

Use **Slime** first because its async rollout architecture is a strong fit for long-running agentic environments.

Primary Slime extension point:

- `--custom-generate-function-path`: implement the per-sample agent loop and Daytona sandbox interaction while keeping Slime's default rollout orchestration.

## Second integration

**Harbor** is the next **framework/backend under the gym facade** (Terminal-Bench / agent-eval), not the primary CLI. Typical buyer shape: BYO GPU + Harbor tasks + Daytona sandboxes — product value is correlated GPU vs sandbox time. See `PRODUCT_DECISIONS.md` §2 / §15 / §16 and `HARBOR_INTEGRATION.md`.

Only use:

- `--rollout-function-path`

if Daytona later needs to replace Slime's outer rollout orchestration entirely.

Reward logic should remain user-configurable, typically through Slime's `--custom-rm-path` or environment-owned reward helpers.

## MVP user experience (goal)

```python
from daytona_gym import (
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
)

config = TrainConfig(
    model=Qwen25_3B(),
    dataset=PromptJsonlDataset("examples/coding_dogfood/prompts/coding_one.jsonl"),
    recipe=Qwen25_3B_Recipe(),
)
run = config.launch(dry_run=True)  # or launch() on a Slime GPU host
print(run.training_run_id, run.inspect_hint)
```

Expected result:

- Training starts on **user-provided** GPU workers (not Daytona-owned compute).
- Daytona provisions coding sandboxes for rollouts.
- Each rollout gets a stable `run_id` and `rollout_id`.
- Telemetry streams automatically; partners open a run view (dashboard later; `dg` today).

Example (aspirational):

```text
✓ connected gpu-worker-01: 1x H100 (BYO)
✓ slime control process started
✓ daytona sandboxes ready
✓ run: run_01J...
✓ inspect: dg stats  |  dashboard: https://app.daytona.io/gym/runs/run_01J...
```

### Current state (honest)

- Slime × Daytona coding dogfood is **proven** on BYO H100 (`examples/coding_dogfood/`, edge suite in `FRICTION.md`).
- **Gym SDK skeleton is shipped:** `TrainConfig(...).launch(dry_run=True|False)` builds (and on a Slime GPU host, runs) the same wiring as the dogfood shell script. See `examples/gym_sdk/quickstart.py`.
- Local dashboard: `dg dash` (or `dg open`) serves HTML over `runs/*.jsonl`.
  - On RunPod/SSH it prints a proxy URL or tunnel one-liner for your laptop.
  - On a laptop: `dg dash --remote user@gpu-host` (or `DAYTONA_GYM_SSH=…`) pulls `runs/` via SSH+tar, then serves.
- Remote BYO worker registration (SSH / agent) is **not shipped** — `launch()` runs on the GPU box itself.

## Core architecture

```text
                           Daytona Control Plane
                         /                       \
                        /                         \
               User GPU Workers             Daytona Sandboxes
              ------------------             -----------------
              Slime                           rollout env #1
              SGLang                          rollout env #2
              Megatron                        rollout env #3
              trainer                         ...
                    \                         /
                     \                       /
                      ---- telemetry/events --
                                |
                                v
                         Daytona Dashboard
```

## What Daytona owns

- sandbox provisioning and lifecycle,
- stateful rollout execution,
- concurrency controls,
- retries and failure metadata,
- rollout-level tracing,
- sandbox/tool timing,
- cross-layer metric correlation,
- run history and dashboards,
- optional GPU control plane later.

## What Daytona does not own in v1

- RL algorithms,
- policy loss implementation,
- gradient computation,
- optimizer implementation,
- model architecture,
- inference engine implementation,
- GPU kernel optimization,
- weight synchronization internals,
- custom reward semantics.

## Design principle

Daytona must stay portable.

A user should be able to move from Lambda, CoreWeave, on-prem, or another GPU provider to Daytona GPUs later without changing the rollout API or losing observability history.

The sticky layer is **workflow + telemetry + debugging**, not compute lock-in.

## Repo documents

- `COMPETITIVE_MODAL_GYM.md` — Modal Training Gym vs us (BYO + Daytona sandboxes).
- `ARCHITECTURE.md` — components, ownership, data flow, identifiers.
- `MVP_SCOPE.md` — explicit v1 scope and non-goals.
- `SLIME_INTEGRATION.md` — how Daytona plugs into Slime.
- `HARBOR_INTEGRATION.md` — Harbor as gym backend (planned).
- `OBSERVABILITY.md` — metrics, traces, dashboard design.
- `CONFIG_AND_CLI.md` — proposed config and command surface.
- `PRODUCT_DECISIONS.md` — important choices and rationale.
- `IMPLEMENTATION_PLAN.md` — suggested build order.
- `AGENTS.md` — coding-agent instructions and constraints.
