# Coding Agent Instructions

## Objective

Implement the Daytona Training Gym MVP described in this repository's markdown specs.

Read in this order:

1. `README.md`
2. `PRODUCT_DECISIONS.md`
3. `ARCHITECTURE.md`
4. `SLIME_INTEGRATION.md`
5. `MVP_SCOPE.md`
6. `OBSERVABILITY.md`
7. `CONFIG_AND_CLI.md`
8. `IMPLEMENTATION_PLAN.md`

## Non-negotiable constraints

- Do not build a new RL trainer.
- Do not fork or modify Slime core unless there is no viable plugin path.
- Prefer Slime `--custom-generate-function-path` for the MVP.
- Keep Daytona core runtime framework-neutral.
- Slime-specific code belongs under an adapter boundary.
- Harbor is the planned second adapter; Harbor/Terminal-Bench types also stay out of core.
- Bring-your-own GPU is a first-class requirement.
- Do not assume Daytona owns the GPU machine.
- Reward semantics remain user-configurable.
- All rollout APIs should support async execution.
- Every rollout must have stable correlation identifiers.
- Cleanup must occur on success, exception, cancellation, and timeout.
- Observability instrumentation must not materially block rollout execution.

## First implementation target

Build only Milestone 0 from `IMPLEMENTATION_PLAN.md` unless explicitly asked to continue.

Create:

- package skeleton,
- framework-neutral rollout/environment interfaces,
- fake environment implementation,
- Slime adapter contract,
- tests.

Do not start dashboard, GPU worker, or production telemetry backend yet.

## Interface preference

Core runtime should look approximately like:

```python
class EnvironmentRuntime(Protocol):
    async def create(self, spec: EnvironmentSpec) -> EnvironmentHandle: ...
    async def execute(self, env: EnvironmentHandle, action: ToolAction) -> ToolResult: ...
    async def reset(self, env: EnvironmentHandle) -> None: ...
    async def close(self, env: EnvironmentHandle) -> None: ...
```

Adapters may translate framework-native objects into Daytona objects.

Do not make `EnvironmentRuntime` accept a Slime `Sample`.

## Testing philosophy

**Real Daytona SDK + real sandboxes are first-class.** Opt-in live tests under
`tests/e2e/` (`pytest tests/e2e --e2e` or with `DAYTONA_API_KEY`) prove behavior
against the platform. Dogfood on BYO GPU is the integration proof for Slime.

**`FakeAsyncDaytona` is second-class.** It exists only so CPU CI can run without
network/API keys. It must mirror **observed** SDK shapes (exception types,
messages, response fields) captured from live e2e / dogfood — never invent
convenient fake behavior that the real SDK does not exhibit. When live and fake
disagree, fix the fake (or the production mapper), and add/adjust a live e2e.

Also keep deterministic core/runtime contract tests (no Daytona) for:

- normal successful rollout,
- tool failure,
- sandbox creation failure,
- timeout,
- cancellation,
- adapter serialization,
- cleanup in all cases,
- concurrency isolation.

`FakeEnvironmentRuntime` (in-process, no SDK) is fine for framework-neutral
rollout/telemetry tests. Do not use it to “prove” Daytona API semantics.

## Error handling

Use typed errors or structured error codes for expected failure classes.

Do not reduce every failure to a generic exception string.

Suggested codes:

```text
sandbox_provision_failed
sandbox_timeout
tool_timeout
tool_failed
rollout_timeout
inference_failed
reward_failed
user_code_error
platform_error
```

## Telemetry hooks

Even before the backend exists, add no-op instrumentation interfaces at major boundaries:

```python
with tracer.span("rollout"):
    ...

with tracer.span("sandbox.provision"):
    ...

with tracer.span("tool.run_tests"):
    ...
```

Make the default tracer cheap/no-op.

## Security

Treat sandbox outputs, repository contents, prompts, model outputs, environment variables, and logs as potentially sensitive.

- never log secrets by default,
- redact environment variables,
- limit captured stdout/stderr,
- do not expose host credentials to sandbox workloads,
- do not give the worker agent destructive host permissions unless required and documented.

## When uncertain

Prefer the smaller product boundary.

Daytona's initial value is:

> portable rollout environments + useful RL observability.

Anything that does not materially support that should probably stay out of the MVP.
