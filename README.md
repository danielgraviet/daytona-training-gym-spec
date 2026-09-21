# Daytona Training Gym — MVP Spec

## Goal

Build a portable reinforcement-learning post-training runtime where users can:

1. bring their own GPUs,
2. use an existing RL framework (Slime first),
3. run agentic rollouts inside Daytona sandboxes,
4. automatically collect rollout + training + inference telemetry,
5. inspect one unified dashboard for bottlenecks and failures.

Daytona should **not** become a new RL training framework.

The product boundary is:

> Existing RL framework owns optimization. Existing inference engine owns token generation. Daytona owns stateful rollout environments, rollout execution infrastructure, and cross-layer observability.

## First integration

Use **Slime** first because its async rollout architecture is a strong fit for long-running agentic environments.

Primary Slime extension point:

- `--custom-generate-function-path`: implement the per-sample agent loop and Daytona sandbox interaction while keeping Slime's default rollout orchestration.

## Second integration

**Harbor** (Terminal-Bench-style harnesses that already configure Daytona sandboxes) is the immediate second adapter. Typical buyer shape: BYO GPU cluster + Harbor fork + Daytona backends — product value is correlated GPU vs sandbox time allocation. See `PRODUCT_DECISIONS.md` §2 / §15 and `HARBOR_INTEGRATION.md`.

Only use:

- `--rollout-function-path`

if Daytona later needs to replace Slime's outer rollout orchestration entirely.

Reward logic should remain user-configurable, typically through Slime's `--custom-rm-path` or environment-owned reward helpers.

## MVP user experience

```bash
pip install daytona-gym

daytona gym init
# edit daytona-gym.yaml

daytona gym run daytona-gym.yaml
```

Expected result:

- Slime starts on user-provided GPU workers.
- Daytona provisions coding sandboxes for rollouts.
- Each rollout gets a stable `run_id` and `rollout_id`.
- Telemetry starts streaming automatically.
- CLI prints a dashboard URL immediately.

Example:

```text
✓ connected gpu-worker-01: 8x H100
✓ slime control process started
✓ rollout runtime connected
✓ dashboard: https://app.daytona.io/gym/runs/run_01J...
```

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

- `ARCHITECTURE.md` — components, ownership, data flow, identifiers.
- `MVP_SCOPE.md` — explicit v1 scope and non-goals.
- `SLIME_INTEGRATION.md` — how Daytona plugs into Slime.
- `OBSERVABILITY.md` — metrics, traces, dashboard design.
- `CONFIG_AND_CLI.md` — proposed config and command surface.
- `PRODUCT_DECISIONS.md` — important choices and rationale.
- `IMPLEMENTATION_PLAN.md` — suggested build order.
- `AGENTS.md` — coding-agent instructions and constraints.
