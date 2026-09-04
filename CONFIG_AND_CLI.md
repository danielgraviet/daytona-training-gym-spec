# Configuration and CLI

## Principle

Do not invent a replacement configuration language for Slime.

Daytona config should describe only Daytona-owned concerns plus references to existing framework configuration.

## Proposed config

`daytona-gym.yaml`

```yaml
version: 1

run:
  name: qwen-coding-grpo
  project: coding-rl

framework:
  type: slime
  config: ./slime.sh

compute:
  mode: byo
  workers:
    - name: gpu-worker-01
      address: gpu-worker-01

rollouts:
  environment: coding
  image: ghcr.io/acme/coding-env:latest
  max_concurrency: 512
  timeout_seconds: 300

  tools:
    - run_command
    - read_file
    - write_file
    - apply_patch
    - run_tests

reward:
  module: ./reward.py
  function: reward

observability:
  enabled: true
  capture_stdout: errors
  capture_model_text: true
  trace_sample_rate: 1.0
```

## Framework config ownership

The user still owns native Slime configuration/arguments.

Example Slime script:

```bash
python train.py \
  ...user's normal slime args... \
  --custom-generate-function-path daytona_gym.adapters.slime.generate \
  --custom-rm-path my_project.reward.reward
```

The Daytona launcher may inject its adapter argument automatically, but should make the final generated command visible.

## CLI

### Authenticate

```bash
daytona login
```

### Initialize project

```bash
daytona gym init
```

Creates:

```text
daytona-gym.yaml
reward.py
.env.example
```

Do not generate unnecessary framework boilerplate if the user already has a Slime project.

### Connect GPU worker

Potential flow:

```bash
daytona gym worker install
```

or a copy/paste install command from the dashboard.

Then:

```bash
daytona gym workers
```

Example output:

```text
NAME            STATUS   GPUS       DRIVER    LOAD
gpu-worker-01   ready    8x H100    580.x     3%
```

### Validate

```bash
daytona gym validate daytona-gym.yaml
```

Checks:

- auth,
- worker reachable,
- GPU visibility,
- Docker/container runtime if required,
- Slime import available,
- SGLang import available,
- Daytona sandbox image accessible,
- reward module imports,
- config schema valid.

### Run

```bash
daytona gym run daytona-gym.yaml
```

Expected output:

```text
Run: run_01J...
Framework: slime
GPU workers: 1 / 1 ready
Rollout environment: coding
Dashboard: https://app.daytona.io/gym/runs/run_01J...
```

### Inspect

```bash
daytona gym status run_01J...
daytona gym logs run_01J...
daytona gym open run_01J...
```

### Stop

```bash
daytona gym stop run_01J...
```

Stopping must terminate Daytona-managed sandboxes and the launched training process, but never destroy the user's GPU host.

## Future compute portability

Later support:

```yaml
compute:
  mode: daytona
  gpu: H100
  count: 8
```

The rest of the file should remain unchanged.

That is a product requirement: **changing GPU provider must not require changing the rollout configuration.**
