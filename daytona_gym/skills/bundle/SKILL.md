# Daytona Training Gym

Launch coding RL on BYO GPUs (RunPod / SSH) with Daytona sandboxes + a live dash.

## Happy path

```bash
export DAYTONA_API_KEY=...
export RUNPOD_API_KEY=...
export RUNPOD_POD_ID=...          # or --create-pod
export DAYTONA_GYM_SSH_IDENTITY=~/.ssh/id_ed25519

python examples/gym_sdk/remote_from_laptop.py --launch --detach
# → run id + dashboard URL, then "Training complete"
```

Create a pod if you do not have one:

```bash
python examples/gym_sdk/remote_from_laptop.py --create-pod --launch --detach
```

## Python SDK

```python
from daytona_gym import (
    TrainConfig, PromptJsonlDataset, Qwen25_3B, Qwen25_3B_Recipe, runpod_worker,
)

run = TrainConfig(
    model=Qwen25_3B(),
    dataset=PromptJsonlDataset("examples/coding_dogfood/prompts/coding_one.jsonl"),
    recipe=Qwen25_3B_Recipe(),
).launch(worker=runpod_worker(), detach=True)
print(run.training_run_id, run.dashboard_url)
run.result()  # Training complete
```

## CLI

```text
dg dash --remote user@ssh.runpod.io -i ~/.ssh/id_ed25519   # durable laptop dash
dg run list
dg run status <id>
dg run logs <id>
dg run wait <id>
dg run stop <id>
dg skills install
```

Prefer `backend="slime"` (default). Harbor is a second adapter under TrainConfig, not a `dg harbor` CLI.
