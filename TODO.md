# TODO — pick up here (next session)

Quick orientation for Daytona Training Gym.

## Status snapshot

**Repo:** `main`  
**Local path:** `~/Desktop/projects/daytona-training-gym-spec`  
**North star:** Modal-shaped gym UX + BYO GPU + Daytona sandboxes — see `COMPETITIVE_MODAL_GYM.md` and `PRODUCT_DECISIONS.md` §16.

### Proven on GPU (H100 dogfood)

| Check | Result |
|---|---|
| Slime train + Daytona custom generate/RM | Pass (0.5B → 3B) |
| Sandbox provision → seed → bootstrap → finalize | Pass |
| Stress / edge suite | Pass (`FRICTION.md` §17–22) |
| `dg ls` / `dg stats` / `dg dash` | Pass (local HTML over JSONL) |

### Gym SDK skeleton (done)

- [x] `TrainConfig` / `CodingRecipe` / `LocalSlimeCompute` / `PromptJsonlDataset` / `TrainingRun`
- [x] `launch(dry_run=True)` builds Ray/Slime argv + runtime_env (no API key in Ray env)
- [x] `launch()` on GPU host: preflight → ray job submit (attached)
- [x] `examples/gym_sdk/quickstart.py` + CPU tests

```bash
python examples/gym_sdk/quickstart.py
# on Slime GPU host:
# DAYTONA_API_KEY=... python examples/gym_sdk/quickstart.py --launch
# dg stats runs/<run_id>.jsonl
```

---

## Next

1. **Dogfood `launch()` on H100 or 3090** — replace bash entry with quickstart `--launch`; note FRICTION.
2. Then pick: Harbor-as-backend under recipes, or polish dashboard (live refresh / hosted URL).

Parked: `dg harbor` CLI wrapper; remote SSH worker agent; PyPI until API stabilizes.

---

## Backlog

### P1

- [ ] Real `TrainConfig.launch()` on BYO GPU (parity with `run_on_slime_pod.sh`)
- [x] Local dashboard v0 (`dg dash` / `dg open` — reward / wall / rollout timeline)
- [ ] Hosted / shared dashboard URL (optional)
- [ ] Remote BYO worker registration

### P1b — Harbor backend (when chosen)

- [ ] Instrument Harbor Daytona path under gym recipes
- [ ] CPU fake-caller contract tests

### P2

- [ ] More recipe presets (model catalog)
- [ ] Detached TrainingRun / wait handles
- [ ] Soft-fail mixture recipes

---

## Key files

| File | Why |
|---|---|
| `daytona_gym/gym/` | TrainConfig facade |
| `examples/gym_sdk/quickstart.py` | Partner entry |
| `COMPETITIVE_MODAL_GYM.md` | Modal vs us |
| `examples/coding_dogfood/` | Proven shell path |
| `daytona_gym/cli.py` | `dg` inspect |

---

*Next: dogfood Gym SDK `--launch` on a BYO GPU box.*
