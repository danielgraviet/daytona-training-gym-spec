# Product Decisions

## 1. Build on existing RL frameworks

Decision:

Do not build a Daytona RL trainer.

Reason:

TRL, Slime, verl, and other frameworks already own optimization and algorithm implementation. Daytona's differentiated infrastructure is environments, rollout execution, and observability.

## 2. Slime first, framework-neutral core

Decision:

Ship Slime as the first adapter.

Reason:

Its async rollout model fits long-running agentic workloads well and it exposes explicit customization hooks for sandbox/tool-based generation.

Constraint:

No Slime type may leak into Daytona's core rollout runtime API.

## 3. Use Slime custom generation before replacing rollout orchestration

Decision:

Start with `--custom-generate-function-path`.

Reason:

Slime already handles async rollout orchestration, buffering, sampling, and surrounding RL system concerns. Replacing the entire rollout function would increase scope without proving product value.

## 4. Bring-your-own GPU first

Decision:

The MVP must work on user-owned or third-party GPU machines.

Reason:

- avoids compute lock-in,
- lets Daytona launch before managed GPUs are mature,
- matches existing research teams with already-allocated clusters,
- lets the product compete on rollout infrastructure rather than GPU pricing.

## 5. Daytona GPUs are an optional backend later

Decision:

Managed Daytona GPU compute should use the same `compute` abstraction as BYO workers.

Reason:

Users should switch compute providers without changing training/rollout code.

## 6. Metrics are a first-class product, not an add-on

Decision:

Observability ships in the MVP.

Reason:

The useful differentiation is not merely provisioning sandboxes. It is explaining the end-to-end critical path between model generation and environment execution.

## 7. Reuse existing metrics

Decision:

Ingest Slime/SGLang/GPU telemetry rather than reimplementing it.

Reason:

Daytona should add unique environment and correlation data, not duplicate mature monitoring sources.

## 8. Open telemetry formats

Decision:

Prefer OpenTelemetry + Prometheus-compatible metrics + structured logs.

Reason:

The product thesis is portability, so telemetry should not create unnecessary lock-in either.

## 9. Coding environments first

Decision:

Optimize the first environment for repository/coding tasks.

Reason:

Coding rollouts naturally require persistent filesystem state, tools, tests, package dependencies, and isolation. This makes Daytona's value obvious.

## 10. Stateful environment != one new VM every tool call

Decision:

A rollout owns a logical environment that persists across actions.

Reason:

The sequence matters:

```text
edit -> test -> inspect -> edit -> test
```

Daytona may implement reset through fresh provisioning, snapshot restore, or reusable environment primitives, but those are runtime choices.

## 11. Reward remains user-owned

Decision:

Provide reward interfaces/helpers but do not hard-code reward semantics.

Reason:

Different RL tasks have fundamentally different verifiers and reward functions.

## 12. Do not require Daytona to proxy all inference

Decision:

SGLang remains the inference engine and may run directly on user GPU infrastructure.

Reason:

Daytona does not need to sit in the token-generation data path to provide value. It needs enough IDs/timings to correlate inference with rollouts.

## 13. Historical observability is the sticky layer

Decision:

Persist run/rollout metadata even when GPUs are external.

Reason:

Users should stay because Daytona becomes the place they debug and compare RL runs, not because moving compute away is painful.

## 14. Avoid premature multi-framework support

Decision:

Build the core with adapter interfaces but ship Slime first.

Reason:

A half-working Slime + TRL + verl integration is worse than an excellent Slime workflow with clean extension boundaries.
