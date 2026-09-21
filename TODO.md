# TODO — pick up here (next session)

Quick orientation for Daytona Training Gym × Slime GPU dogfood.

## Status snapshot

**Repo:** `main` @ `f97d2a9`+ (bootstrap via Ray `runtime_env`; docs updated)  
**GPU:** RunPod A100 pod with `slimerl/slime:latest` (tear down if idle — check billing)  
**Local path:** `~/Desktop/projects/daytona-training-gym-spec`  
**Pod path:** `/root/daytona-training-gym-spec`, Slime `/root/slime`, Megatron `/root/Megatron-LM`

### Proven on GPU

| Check | Result |
|---|---|
| Slime 1-step train (0.5B / 1.5B) | Pass |
| Daytona custom generate + RM | Pass |
| Sandbox provision → seed → finalize | Pass |
| Concurrency=2 cleanup | Pass |
| Preflight bad API key (fail-fast) | Pass (`DaytonaAuthenticationError`) |
| Coding seed files | Pass |
| Bootstrap `run_tests` | **Pass** — `raysubmit_qMmVkw1ZeGapWCCi` (`tool.run_tests` before generate) |
| Model-emitted JSON tools / `write_file` | Not yet (0.5B + 1.5B ignore tool JSON) |

### Important pod state

- Models on disk: `Qwen2.5-0.5B-Instruct` (+ `_torch_dist`), `Qwen2.5-1.5B-Instruct` (+ `_torch_dist`)
- Slime model script patch: `/root/slime/scripts/models/qwen2.5-1.5B.sh` → `--rotary-base 1000000` (was 10000)
- Launch script may still be sed’d to 1.5B paths on the pod
- **Rotate `DAYTONA_API_KEY`** — leaked in shell history, chat, and `ray job list` runtime_env

---

## First 10 minutes next session

1. **Pod still alive?** If yes and you want more GPU tests, SSH in. If not, skip to CPU / P1.
2. `cd /root/daytona-training-gym-spec && git pull && pip install -e .`
3. `export DAYTONA_API_KEY=...` (prefer a **new** rotated key)  
   `export DAYTONA_API_URL=https://app.daytona.io/api`
4. `python -m daytona_gym.preflight`
5. Optional sanity: `python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0`

---

## Next tasks (priority order)

### P0 — Done (2026-09-18)

- [x] Bootstrap `tool.run_tests` on GPU (`raysubmit_qMmVkw1ZeGapWCCi`)
- [x] Wire bootstrap/seed through Ray `runtime_env` + `generate.py` env resolve (`f97d2a9`)
- [x] Update `FRICTION.md` / `GPU_RUN.md`

### P1 — Product / DX follow-ups (CPU OK)

- [ ] Redact or warn: Ray `runtime_env` dumps API keys in `ray job list`
- [ ] Optional: commit parameterized `run_on_slime_pod.sh` knobs for `MODEL_SCRIPT` / `HF_CHECKPOINT` if pod still uses sed
- [ ] Add short “partner dogfood” blurb pointing at `examples/coding_dogfood/README.md` + preflight

### P1b — Harbor as second adapter (product track)

Decision locked: Slime = MVP adapter; Harbor = immediate second (`PRODUCT_DECISIONS.md` §2 / §15, `HARBOR_INTEGRATION.md`).

- [ ] Spike Harbor’s Daytona sandbox config / extension points (read-only research)
- [ ] Sketch `daytona_gym/adapters/harbor/` boundary (no Harbor types in core)
- [ ] Define partner demo: BYO GPU + Harbor + Daytona timeline (GPU vs sandbox wall-time)
- [ ] CPU contract test with fake Harbor-shaped caller before any GPU Harbor dogfood

### P2 — Real model tool loop (needs GPU + time)

- [ ] Try Qwen2.5-**4B** (or stronger instruct) for model-emitted `write_file` + second `run_tests` → `reward=1.0`
- [ ] Or scripted/forced multi-step bootstrap (seed → fail tests → inject fix hint) — only if product wants it

### P3 — Learning / narrative

- [ ] Skim `docs/SLIME_STACK_LEARNING.md` before partner calls
- [ ] Skim `examples/coding_dogfood/FRICTION.md` for war stories

---

## Key files

| File | Why |
|---|---|
| `examples/coding_dogfood/run_on_slime_pod.sh` | One-command GPU launch (+ Ray env for bootstrap) |
| `examples/coding_dogfood/FRICTION.md` | What hurt / what worked |
| `examples/coding_dogfood/GPU_RUN.md` | Pass criteria checklist |
| `docs/SLIME_STACK_LEARNING.md` | SGLang / Megatron / Slime talk track |
| `daytona_gym/preflight.py` | Fail-fast Daytona auth |
| `daytona_gym/adapters/slime/generate.py` | Env-resolved seed + bootstrap for Ray workers |
| `daytona_gym/adapters/slime/generate_dogfood.py` | Seed + bootstrap defaults |
| `daytona_gym/runtime/rollout.py` | Seed + bootstrap + agent loop |
| `HARBOR_INTEGRATION.md` | Second adapter plan (Harbor / Terminal-Bench) |
| `SLIME_INTEGRATION.md` | Product integration decisions |
| `PRODUCT_DECISIONS.md` | Slime first → Harbor second; BYO GPU; metrics wedge |

---

## Commands cheat sheet

```bash
# Local
cd ~/Desktop/projects/daytona-training-gym-spec
uv run python -m pytest -q
uv run python scripts/validate_dogfood_path.py

# Pod — coding dogfood
export DAYTONA_API_KEY=...
python -m daytona_gym.preflight
bash examples/coding_dogfood/run_on_slime_pod.sh
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0

# Disable bootstrap if needed
DAYTONA_BOOTSTRAP_RUN_TESTS=0 bash examples/coding_dogfood/run_on_slime_pod.sh
```

---

## Done this session (don’t redo)

- Slime+Daytona end-to-end train on A100  
- Seed files, concurrency=2, preflight, loud generate failures  
- Bootstrap `run_tests` implementation + **GPU verify** (`qMmVkw1ZeGapWCCi`)  
- Ray `runtime_env` path for bootstrap/seed (args setattr alone was flaky)  
- Learning doc for Slime/SGLang/Megatron  
- Friction / GPU_RUN updated through bootstrap success  

---

*Next: Harbor spike (P1b) on CPU, or P2 4B tool loop / tear down the pod.*
