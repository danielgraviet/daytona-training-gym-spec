# Competitive brief: Modal Training Gym vs Daytona Training Gym

Audience: product + engineering. Goal: shared understanding before the next build spike.
Sources: [gym.modal.dev](https://gym.modal.dev/), [modal-projects/training-gym](https://github.com/modal-projects/training-gym), dashboard guide.

## One-line contrast

| | Modal Training Gym | Daytona Training Gym (ours) |
|---|---|---|
| Partner asks for | “RL post-training that just works” | Same UX shape |
| Who owns GPUs | **Modal** | **BYO** (RunPod, 3090, cluster, …) |
| Who owns sandboxes | **Modal Sandbox** | **Daytona** sandboxes |
| Sticky layer | Modal app + dashboard + recipes | Workflow + correlated env/GPU traces + (later) dashboard |

We are **not** competing by wrapping `harbor run` flags. We compete by matching Modal’s **gym SDK + observability** while keeping compute portable and sandboxes on Daytona.

## What Modal ships (public surface)

### Install / setup

```bash
uv pip install git+https://github.com/modal-projects/training-gym.git@main
modal setup
training-gym setup          # deploy workspace dashboard (Modal App)
training-gym skills install # agent skill bundle for coding assistants
training-gym open           # open dashboard
```

### Partner code path

Partners do **not** assemble Ray jobs by hand. They write:

```python
from modal_training_gym import (
    HuggingFaceDataset,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    TrainConfig,
)

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=HuggingFaceDataset(...),
    recipe=Qwen3_5_4B_Recipe(custom_rm_function=...),
)
run = config.launch()
print(run.training_run_id)
```

`launch()` starts a **detached** Modal training app and returns a `TrainingRun` handle (ids + Modal URLs). Closing the notebook does not kill training.

### What Modal hides (on purpose)

- Cluster topology / multi-node layout
- Ray + NCCL bring-up
- Volume mounts and checkpoint paths
- Weight sync / offload between rollout and train engines
- Serving for eval after train
- Recipe presets per model family (Qwen3/3.5/…, DeepSeek, Gemma, …)

Frameworks sit **under** recipes. Public tree today emphasizes **Slime** (and Miles). Harbor appeared in earlier Training Gym commits as a framework package; the partner-facing story remains `TrainConfig` + recipes, not memorizing Harbor CLI.

### Dashboard (why buyers care)

`training-gym setup` deploys an observability app. Surfaces called out in docs:

1. **Run list** — status + live progress across the workspace
2. **Per run** — mean reward, score/advantage distributions, link to Modal app
3. **Step wall-time breakdown** — generate rollouts, offload, logprobs, train, optimizer, checkpoint, weight sync (built-in profiler)
4. **Per rollout** — prompt / turns / thinking / reward; reward histogram health checks

Custom phases (e.g. custom reward) show up as timeline markers.

### Sandbox / coding RL path

Modal’s sandboxes tutorial shape (Slime + sandboxed verification):

- Train/inference on Modal GPUs via Slime recipe
- Reward / verify inside **Modal sandboxes** (helpers akin to `score_in_sandbox` / Harbor-style task verify)
- Partner still launches via `TrainConfig`, not via raw sandbox provider CLI

That is the pattern buyers mean by “Training Gym”: **one Python object launches train + env-backed reward**, with a place to stare at failures.

## What we already proved (Daytona)

On BYO H100 + cloud Daytona sandboxes (`examples/coding_dogfood/`):

| Capability | Status |
|---|---|
| Slime `--custom-generate-function-path` → Daytona sandbox tool loop | Proven |
| Custom RM / coding seed / bootstrap | Proven |
| Concurrent sandboxes (e.g. 48/48 reward=1 concurrency storm) | Proven |
| Typed failures, timeouts, tool caps, context soft-abort | Proven |
| Local telemetry JSONL + `dg` / `dg ls` / `dg stats` / `dg dash` | Proven |
| `TrainConfig.launch()` facade | **Skeleton shipped** (`daytona_gym.gym`; see `examples/gym_sdk/`) |
| Live run dashboard (Modal-parity) | Local `dg dash` over JSONL; hosted URL **not** shipped |
| BYO worker registration / one-button remote launch | **Not shipped** (manual pod scripts today) |
| Harbor as gym backend | **Not shipped** (spec only) |

Bottom line: the **hard env loop** works. The **Modal-shaped product facade** does not.

## Gap map (to match Modal UX without copying Modal lock-in)

| Modal piece | Daytona equivalent | Gap |
|---|---|---|
| `TrainConfig(...).launch()` | `daytona_gym.gym` facade over Slime dogfood | **Skeleton shipped** (`dry_run` + local `launch()`); remote worker later |
| Modal GPUs | BYO GPU host | Need clear worker/launch story (scripts → worker → SDK) |
| Modal Sandbox | Daytona `EnvironmentRuntime` | Done for Slime path |
| Recipe presets | Dogfood scripts + env knobs | Need named recipes (model + batch + sandbox profile) |
| `training-gym setup/open` dashboard | `dg dash` / `dg open` over `runs/*.jsonl` | Hosted/shared URL still missing |
| `TrainingRun` handle | `run_id` in telemetry | Need stable public run object + inspect URL |
| Agent skills | None | Optional later |
| Harbor under gym | `HARBOR_INTEGRATION.md` | Backend/adapter under recipes — **not** partner CLI |

## Explicit non-copy / non-goals

From our product stance (unchanged):

- Do **not** build a new RL trainer (Slime/Harbor/etc. own algorithms).
- Do **not** require Daytona-owned GPUs for MVP.
- Do **not** make “type long `harbor run -e module:Class --plugin …`” the happy path.
- Do **not** fork Harbor; deepen its Daytona path when we integrate it.

## Product implication

**North star:** Modal-shaped gym (`TrainConfig` / recipes / run handle / dashboard) with **BYO GPUs** and **Daytona sandboxes**.

**Harbor’s role:** second framework/backend under that gym (Terminal-Bench / agent-eval distribution), instrumented for GPU↔sandbox correlation — not the primary partner interface.

## Next build fork

1. **Dogfood Gym SDK `launch()` on BYO GPU** — prove `examples/gym_sdk/quickstart.py --launch` matches shell dogfood.
2. **Harbor-as-backend** under the same gym facade (still no Harbor-CLI-first UX), or polish local dashboard → hosted URL.

## Related repo docs

- `PRODUCT_DECISIONS.md` — §16 Modal-shaped UX; §2 / §15 Harbor as backend
- `HARBOR_INTEGRATION.md` — adapter spike (deferred under gym facade)
- `examples/coding_dogfood/` — proven Slime×Daytona path
- `MVP_SCOPE.md` / `OBSERVABILITY.md` — dashboard success criteria
