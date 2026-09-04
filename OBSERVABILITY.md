# Observability

## Product thesis

The dashboard should answer:

> Why is this RL run slow, expensive, unstable, or failing?

Do not stop at generic GPU utilization charts.

Daytona's differentiation is correlating:

- trainer state,
- inference state,
- GPU resource utilization,
- rollout scheduling,
- sandbox lifecycle,
- tool execution,
- reward/verifier execution.

## Reuse existing telemetry

Do not rebuild every metric source.

### Slime

Consume/export its training metrics such as:

- reward,
- loss,
- policy metrics,
- evaluation metrics,
- rollout step metrics.

### SGLang

Scrape serving metrics and/or consume request timing metadata:

- queue depth,
- running requests,
- end-to-end request latency,
- queue time,
- generation throughput,
- cache/serving metrics where available.

### GPU worker

Collect:

- GPU utilization,
- GPU memory used/total,
- power draw if available,
- GPU temperature if useful,
- host CPU/memory,
- process liveness.

### Daytona

Instrument the rollout/environment layer directly.

## Canonical trace

Each rollout should be one distributed trace.

```text
rollout
├── sandbox.provision
├── inference.generate
├── tool.run_command
│   └── pytest
├── inference.generate
├── tool.apply_patch
├── tool.run_tests
├── reward.compute
└── sandbox.finalize
```

## Required span attributes

Every relevant span should carry:

```text
project_id
run_id
rollout_id
sample_id
sandbox_id
worker_id
training_step
rollout_batch_id
```

Inference spans should additionally carry a `model_request_id` if one is available.

## Core derived metrics

### Rollout metrics

- rollouts completed / minute,
- active rollouts,
- queued rollouts,
- p50/p95/p99 rollout duration,
- completed/aborted/failed/truncated counts,
- average tool calls per rollout,
- average inference calls per rollout.

### Environment metrics

- sandbox provision p50/p95/p99,
- sandbox reset p50/p95/p99,
- tool execution p50/p95/p99 by tool,
- environment CPU time,
- environment wall time,
- sandbox failure rate,
- timeout rate.

### Inference metrics

- inference queue time,
- time-to-first-token if available,
- generation latency,
- tokens/sec,
- active/waiting requests.

### Training metrics

- loss,
- reward,
- policy update time,
- training step duration,
- optimizer step time where available.

### GPU metrics

- utilization,
- memory utilization,
- idle time,
- idle intervals.

## Daytona-specific derived metrics

These should become flagship metrics.

### 1. Rollout wall-time decomposition

For a run:

```text
Inference waiting       18%
Inference generation    21%
Sandbox provisioning     4%
Environment execution   46%
Reward/verifier           8%
Other                     3%
```

### 2. Straggler tax

Initial definition:

> Extra wall-clock time caused by the tail of rollout durations relative to a chosen baseline (for example median or scheduler-ready threshold).

Do not pretend this has one universally correct formula. Display definition in UI.

### 3. Environment-bound vs inference-bound

Classify a run based on observed critical-path time.

Example:

```text
Environment bound: 57% of rollout critical-path time is sandbox/tool execution.
```

### 4. GPU rollout-starvation time

Estimate intervals where training/inference GPUs are underutilized while the system is waiting for rollout supply.

Be careful with causality. Label this an estimate unless the scheduler gives a definitive reason.

### 5. Waste from failed rollouts

Track:

- GPU generation time spent on eventually failed rollouts,
- sandbox time spent on failed rollouts,
- retries per successful sample.

## Run overview dashboard

Above the fold:

```text
Run status       RUNNING
Reward           0.47 -> 0.63
GPU utilization  72%
Rollouts/min      238
p95 rollout       39.8s
Failure rate      0.7%
```

Then:

1. reward/loss over time,
2. GPU utilization + rollout throughput on same time axis,
3. rollout latency distribution,
4. wall-time decomposition,
5. slowest tool types,
6. failure categories.

## Individual rollout page

Show a chronological trace:

```text
00.000  rollout start
00.013  sandbox request
00.811  sandbox ready
00.824  inference requested
02.122  inference completed
02.130  run_tests
14.880  run_tests completed
14.899  inference requested
...
23.510  reward computed
23.540  rollout complete
```

Include:

- prompt/task,
- tool calls,
- stdout/stderr excerpts,
- exit codes,
- model responses,
- reward,
- final status,
- sandbox metadata.

Redact secrets by default.

## Implementation recommendation

Use open standards underneath:

- OpenTelemetry for traces,
- Prometheus-compatible metrics,
- structured logs.

Daytona may store/index these centrally, but the instrumentation should not require a proprietary protocol at every layer.

## Telemetry overhead target

Observability must not materially slow rollouts.

Design for:

- batched event export,
- asynchronous export,
- sampling for verbose logs,
- full metrics but optional full payload capture,
- configurable retention.

## Existing Slime observability

Slime already sends training metrics to Weights & Biases / TensorBoard and exposes SGLang request/performance information. Daytona should ingest/correlate these rather than duplicating them.

Slime also has sample traces/debug rollout data. Use these where useful, but Daytona should instrument sandbox spans because Slime cannot infer sandbox internals automatically.
