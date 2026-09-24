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

### Next

1. ~~Dogfood `launch()` on H100~~ — **done** (Gym SDK on RunPod).
2. ~~Wire `TrainingRun` → live dash URL~~ — **done** (`launch(open=True)` / `run.open()`).
3. ~~BYO `SshWorker`~~ — **done** (laptop → RunPod **proxy PTY** + direct TCP when sshd exists).
4. ~~Detached `TrainingRun`~~ — **done** (`launch(detach=True)` / `--detach` → run id + dash URL; train continues on worker).
5. Pick one:
   - **RunPod create-pod sugar** (`runpod_worker` that creates+waits, not just resolves).
   - **Outbound AgentWorker** (pod dials out; no inbound SSH at all).
   - Harbor-as-backend when chosen.

Parked: `dg harbor` CLI wrapper; PyPI until API stabilizes.

---

## Backlog

### P1

- [x] Real `TrainConfig.launch()` on BYO GPU (parity with dogfood; RunPod verified)
- [x] Local dashboard v0 (`dg dash` / `dg open` — reward / wall / rollout timeline)
- [x] `TrainingRun.open()` / `launch(open=True)` → live tunnel URL
- [ ] Hosted Daytona dashboard URL (drop Cloudflare dependency)
- [x] Remote BYO worker via ``SshWorker`` (RunPod proxy PTY + direct TCP)
- [ ] Outbound AgentWorker (no inbound SSH)
- [ ] Optional provider helpers (e.g. RunPod create → SshWorker)

### P1b — Harbor backend (when chosen)

- [ ] Instrument Harbor Daytona path under gym recipes
- [ ] CPU fake-caller contract tests

### P2

- [ ] More recipe presets (model catalog)
- [x] Detached TrainingRun / wait handles (`launch(detach=True)` → run id + dash URL)
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
