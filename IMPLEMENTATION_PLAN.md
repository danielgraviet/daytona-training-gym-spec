# Implementation Plan

## Milestone 0 — spike the Slime hook

Goal: prove one Slime sample can execute against a Daytona sandbox.

Build:

1. minimal Python package `daytona_gym`,
2. Slime `custom_generate` implementation,
3. create one Daytona sandbox,
4. execute one command,
5. feed result back into a simple agent loop,
6. return a valid Slime sample.

Do not build dashboard first.

Exit criteria:

- local contract test passes,
- one real rollout completes,
- sandbox always cleans up.

## Milestone 1 — concurrent rollout runtime

Goal: make the adapter safe for many asynchronous rollouts.

Build:

- async Daytona sandbox client,
- concurrency limiter,
- rollout IDs,
- timeouts,
- cancellation,
- retry policy,
- structured failure reasons.

Exit criteria:

- 100+ synthetic concurrent rollout tasks can execute,
- one slow/failing task does not block unrelated tasks,
- no leaked sandboxes after test suite.

## Milestone 2 — minimal telemetry

Goal: create the data model before creating UI.

Instrument spans:

- rollout,
- sandbox.provision,
- inference.generate,
- tool.*,
- reward.compute,
- sandbox.finalize.

Metrics:

- rollout duration,
- sandbox startup,
- tool duration,
- status counts.

Exit criteria:

- one rollout can be reconstructed chronologically from stored telemetry.

## Milestone 3 — BYO GPU worker

Goal: launch/observe Slime on an external GPU box.

Build:

- worker registration,
- GPU inventory,
- process launch,
- stdout/stderr streaming,
- NVIDIA utilization/memory collection,
- safe stop.

Exit criteria:

- user can launch Slime remotely without Daytona owning the GPU machine.

## Milestone 4 — dashboard v0

Goal: answer "what is happening?" and "what is slow?"

Pages:

1. run overview,
2. rollout list,
3. rollout trace.

Charts:

- reward/loss,
- GPU utilization,
- rollout throughput,
- rollout latency distribution,
- wall-time decomposition,
- failures.

Exit criteria:

- slow sandbox/test step can be identified from UI without grepping logs.

## Milestone 5 — CLI + config polish

Goal: new-user workflow.

Implement:

```bash
daytona gym init
daytona gym validate
daytona gym run
daytona gym status
daytona gym open
daytona gym stop
```

Exit criteria:

A user unfamiliar with internal Daytona architecture can run the documented happy path.

## Milestone 6 — performance experiments

Only after functional MVP.

Measure:

- scaling sandbox concurrency,
- p95/p99 environment startup,
- long-tail rollout behavior,
- GPU idle intervals,
- telemetry overhead,
- effects of snapshots/prepared environments,
- retry overhead.

## Suggested package structure

```text
daytona-gym/
├── daytona_gym/
│   ├── cli/
│   ├── config/
│   ├── runtime/
│   │   ├── environment.py
│   │   ├── rollout.py
│   │   └── errors.py
│   ├── adapters/
│   │   └── slime/
│   │       ├── generate.py
│   │       ├── reward.py
│   │       └── sample.py
│   ├── telemetry/
│   │   ├── traces.py
│   │   ├── metrics.py
│   │   └── ids.py
│   └── workers/
│       └── gpu_agent.py
├── tests/
│   ├── contracts/
│   ├── integration/
│   └── concurrency/
├── examples/
│   └── coding_grpo/
└── pyproject.toml
```

## First coding task

The first PR should **not** attempt the whole product.

Implement:

> A CPU-testable Slime adapter contract plus a fake in-memory Daytona environment runtime.

Why:

It forces clean boundaries before external GPU, sandbox, and telemetry complexity is introduced.

The second PR can swap the fake environment for the real Daytona SDK.
