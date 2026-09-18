# TODO — pick up here (2026-09-18 → tomorrow)

Quick orientation for Daytona Training Gym × Slime GPU dogfood.

## Status snapshot

**Repo:** `main` @ `fd2a3e9` (clean, pushed to origin)  
**GPU:** RunPod A100 pod with `slimerl/slime:latest` (may still be up — check before spending $)  
**Local path:** `~/Desktop/projects/daytona-training-gym-spec`  
**Pod path:** `/root/daytona-training-gym-spec`, Slime `/root/slime`, Megatron `/root/Megatron-LM`

### Proven on GPU

| Check | Result |
|---|---|
| Slime 1-step train (0.5B) | Pass |
| Daytona custom generate + RM | Pass |
| Sandbox provision → seed → finalize | Pass |
| Concurrency=2 cleanup | Pass |
| Preflight bad API key (fail-fast) | Pass (`DaytonaAuthenticationError`) |
| Coding seed files | Pass |
| Bootstrap `run_tests` (code shipped) | **Was running on pod when we stopped — verify tomorrow** |
| Model-emitted JSON tools / `write_file` | Not yet (0.5B + 1.5B ignore tool JSON) |

### Important pod state

- Models on disk: `Qwen2.5-0.5B-Instruct` (+ `_torch_dist`), `Qwen2.5-1.5B-Instruct` (+ `_torch_dist`)
- Slime model script patch: `/root/slime/scripts/models/qwen2.5-1.5B.sh` → `--rotary-base 1000000` (was 10000)
- Launch script may still be sed’d to 1.5B paths on the pod
- **Rotate `DAYTONA_API_KEY`** — leaked in shell history, chat, and `ray job list` runtime_env

---

## First 10 minutes tomorrow

1. **Pod still alive?** If yes and you want more GPU tests, SSH in. If not, skip to “Without GPU”.
2. `cd /root/daytona-training-gym-spec && git pull && pip install -e .`
3. `export DAYTONA_API_KEY=...` (prefer a **new** rotated key)  
   `export DAYTONA_API_URL=https://app.daytona.io/api`
4. `python -m daytona_gym.preflight`
5. Check last dogfood run (bootstrap):
   ```bash
   python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl --rollout rollout_0
   # look for: sandbox.seed → tool.run_tests → inference.generate → sandbox.finalize
   # and reward=0.0 (not None) in: ray job logs … | grep daytona-dogfood
   ```

---

## Next tasks (priority order)

### P0 — Finish bootstrap verification (GPU if available)

- [ ] Pod must be on `fd2a3e9`+ (`git log -1`, `grep bootstrap generate_dogfood.py`)
- [ ] Confirm inspect timeline has **`tool.run_tests` before `inference.generate`**
- [ ] Confirm logs show `[daytona-dogfood] start bootstrap='python test_broken.py'`
- [ ] Confirm `[daytona-dogfood] done … reward=0.0` (not None)
- [ ] Confirm Megatron train step still succeeds with bootstrap
- [ ] Update `FRICTION.md` / `GPU_RUN.md` with verified job id

**Note (2026-09-18 eve):** job `raysubmit_dWWbjE8P9CU4XEui` still showed seed→generate with no tools (5 spans) — almost certainly **old code on pod** (no pull / no reinstall). Re-pull and re-run before declaring bootstrap broken.

### P1 — Product / DX follow-ups (CPU OK)

- [ ] Redact or warn: Ray `runtime_env` dumps API keys in `ray job list`
- [ ] Optional: commit parameterized `run_on_slime_pod.sh` knobs for `MODEL_SCRIPT` / `HF_CHECKPOINT` if pod still uses sed (partially in tree — confirm docs match)
- [ ] Add short “partner dogfood” blurb pointing at `examples/coding_dogfood/README.md` + preflight

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
| `examples/coding_dogfood/run_on_slime_pod.sh` | One-command GPU launch |
| `examples/coding_dogfood/FRICTION.md` | What hurt / what worked |
| `examples/coding_dogfood/GPU_RUN.md` | Pass criteria checklist |
| `docs/SLIME_STACK_LEARNING.md` | SGLang / Megatron / Slime talk track |
| `daytona_gym/preflight.py` | Fail-fast Daytona auth |
| `daytona_gym/adapters/slime/generate_dogfood.py` | Seed + bootstrap defaults |
| `daytona_gym/runtime/rollout.py` | Seed + bootstrap + agent loop |
| `SLIME_INTEGRATION.md` | Product integration decisions |

---

## Commands cheat sheet

```bash
# Local
cd ~/Desktop/projects/daytona-training-gym-spec
uv run python -m pytest -q
uv run python scripts/validate_dogfood_path.py

# Pod — coding dogfood (1.5B example)
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
- Bootstrap `run_tests` implementation + tests (83 green)  
- Learning doc for Slime/SGLang/Megatron  
- Friction log updated through late auth failure + rotary_base mismatch  

---

*Update this file when P0 is checked off or the pod is torn down.*
