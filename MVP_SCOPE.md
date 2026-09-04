# MVP Scope

## MVP thesis

Prove that Daytona can make agentic RL post-training easier to run and easier to understand without requiring users to move their GPU compute to Daytona.

## v1 happy path

One supported workflow:

> Slime + SGLang + coding-agent rollouts + bring-your-own NVIDIA GPUs + Daytona sandboxes + Daytona dashboard.

## Must-have capabilities

### Bring-your-own GPU

- Connect one or more Linux GPU hosts.
- Detect GPU inventory.
- Launch Slime training process.
- Collect NVIDIA GPU utilization/memory metrics.
- Preserve provider neutrality.

### Slime integration

- Use Slime's custom generation hook for per-sample agentic rollout execution.
- Preserve Slime's default async rollout scheduler.
- Return a valid `Sample` with required training fields.
- Allow custom reward hook.

### Daytona coding environment

Support a minimal action set:

- `run_command`
- `read_file`
- `write_file`
- `apply_patch`
- `run_tests`

Sandbox lifecycle:

- create from image/snapshot,
- reset,
- execute tools,
- terminate,
- capture resource + timing metadata.

### Observability

At minimum:

- GPU utilization,
- GPU memory utilization,
- rollout throughput,
- rollout p50/p95/p99 duration,
- sandbox provision p50/p95/p99,
- environment execution time,
- inference wait + generation time,
- reward/grading time,
- failed/aborted/truncated rollout counts,
- tool execution distributions,
- individual rollout timeline.

### Run history

Persist run metadata so the user can compare runs later.

## Explicit non-goals

Do not build these for v1:

- a new RL algorithm library,
- a new trainer,
- a new inference engine,
- a new distributed training framework,
- a new experiment tracking product for all ML workloads,
- a generic agent framework,
- support for every Slime recipe,
- support for every cloud provider through native APIs,
- automatic multi-node training topology optimization,
- custom weight synchronization,
- Daytona-managed GPU scheduling,
- perfect cost attribution,
- visual reward-function builder.

## Supported initial task shape

Coding/repository tasks are ideal because they demonstrate why a stateful sandbox matters.

Example rollout:

```text
prompt: fix failing parser tests
  -> inspect repository
  -> run tests
  -> edit source
  -> rerun tests
  -> inspect failure
  -> edit source
  -> rerun tests
  -> final answer
  -> verifier reward
```

## Success criteria

The MVP is successful if a new user can:

1. connect a GPU host,
2. point to model/data/reward/environment config,
3. start a Slime run,
4. see Daytona environments created automatically,
5. observe live rollout metrics,
6. open a slow/failing rollout and understand why it was slow/failing,
7. complete training without Daytona owning the GPU hardware.

## What to benchmark

Do not optimize benchmark numbers before the full path works.

Once functional, measure:

- environments provisioned per second,
- concurrent active sandboxes,
- rollout throughput,
- rollout latency distribution,
- GPU idle fraction,
- GPU idle time attributable to environment waits,
- straggler tax,
- retry/failure overhead,
- telemetry overhead.
