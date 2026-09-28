# Repo-scale task pack

Small multi-file Python package used for Phase 4.3 dogfood.

```python
from daytona_gym import HarborDataset, TrainConfig, Qwen25_05B, Qwen25_05B_Recipe

TrainConfig(
    model=Qwen25_05B(),
    dataset=HarborDataset(path="examples/gym_sdk/repo_pack", train_size=1),
    recipe=Qwen25_05B_Recipe(
        # Optional: warm sandbox from a Daytona snapshot (skips cold image pull).
        # snapshot="daytona-gym-calc-pack",
    ),
)
```

Provision p50/p95 with vs without `recipe.snapshot` show up under
`sandbox_provision` in `dg stats` / the dashboard.
