# Architecture

## System boundary

Daytona Training Gym is a control + rollout + observability layer around existing RL frameworks.

For the initial Slime integration:

```text
Slime
├── Megatron: training / optimization
├── SGLang: inference / token generation
└── rollout loop
      |
      v
Daytona rollout adapter
      |
      +--> Daytona sandbox API
      +--> telemetry SDK
      +--> user reward hook
```

## Main components

### 1. Daytona CLI

Responsibilities:

- authenticate user,
- load config,
- validate GPU worker connectivity,
- create run in Daytona control plane,
- launch Slime command/process,
- inject Daytona Slime adapter paths,
- print dashboard URL,
- stream high-level run status.

### 2. Daytona GPU Worker Agent

Installed on bring-your-own GPU hosts.

Responsibilities:

- register machine + GPU inventory,
- launch user-approved training processes,
- report process lifecycle,
- collect GPU/system metrics,
- expose logs to Daytona control plane,
- maintain outbound-only authenticated connection when possible.

Do not tightly couple this agent to Slime. It should later support TRL and other frameworks.

### 3. Slime Adapter

Initial implementation should integrate through Slime's per-sample custom generation hook.

Responsibilities:

- receive Slime `Sample`,
- assign/propagate `rollout_id`,
- provision/reset Daytona environment,
- call model generation through Slime/SGLang flow,
- execute requested tools in sandbox,
- append tool observations back into agent context,
- repeat until completion,
- attach reward/status/metadata,
- emit trace events,
- return valid Slime `Sample` object.

### 4. Daytona Rollout Runtime

Framework-agnostic logical API.

Suggested internal interface:

```python
class RolloutRuntime:
    async def create_environment(self, spec) -> EnvironmentHandle: ...
    async def execute_tool(self, env, call) -> ToolResult: ...
    async def reset_environment(self, env) -> None: ...
    async def finalize(self, env) -> RolloutEnvironmentSummary: ...
```

The Slime adapter should depend on this API, not directly on low-level sandbox implementation details.

### 5. Daytona Sandbox Runtime

Responsibilities:

- create isolated sandbox,
- start from image/snapshot,
- persistent filesystem + process state during rollout,
- execute commands/tools,
- capture stdout/stderr/exit codes,
- enforce resource/time limits,
- reset/terminate sandbox,
- expose lifecycle timing.

### 6. Telemetry Collector

Receives:

- Daytona rollout spans,
- Daytona sandbox metrics,
- SGLang serving metrics,
- Slime training metrics,
- GPU host metrics.

Normalize around stable identifiers:

```text
project_id
run_id
worker_id
training_step
rollout_batch_id
rollout_id
sample_id
sandbox_id
model_request_id
```

These IDs are critical. Without them, cross-layer correlation will be weak.

### 7. Dashboard

Three levels:

1. run overview,
2. rollout distribution / bottlenecks,
3. individual rollout trace.

## Data flow

```text
prompt/task
    |
    v
Slime rollout scheduler
    |
    v
Daytona custom_generate(sample)
    |
    +--> provision sandbox
    |
    +--> request generation from SGLang
    |
    +--> parse tool/action
    |
    +--> execute in sandbox
    |
    +--> return observation to model
    |
    +--> repeat
    |
    +--> calculate/attach reward
    |
    v
completed Slime Sample
    |
    v
Slime trainer
    |
    v
loss -> gradients -> weight update
```

## State ownership

### Slime owns

- training batch construction,
- optimization loop,
- rollout scheduling semantics,
- training sample consumption,
- model weight lifecycle.

### SGLang owns

- inference request execution,
- token generation,
- serving queues,
- inference cache behavior.

### Daytona owns

- environment state,
- sandbox lifecycle,
- tool execution,
- rollout environment timings,
- cross-system observability.

## Async principle

Do not introduce a Daytona barrier that forces all rollouts to finish together.

The runtime must support many independent in-flight rollouts:

```text
rollout 1: generate -> test -----------> generate -> done
rollout 2: generate -> grep -> done
rollout 3: generate -> pytest -----------------------> done
rollout 4: generate -> edit -> test -> generate -> done
```

Slow environments should be observable and independently cancellable/retryable.

## Framework portability

The core rollout runtime and telemetry schema must not expose Slime-specific types.

Use adapters:

```text
Harbor adapter --\
                 \
Slime adapter ----> Daytona Rollout Runtime -> Daytona Sandboxes
                 /
TRL adapter ----/   (later)
```

Slime is the first adapter (MVP). Harbor is the immediate second. Neither is the architecture.
