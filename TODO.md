# TODO — pick up here (next session)

Quick orientation for Daytona Training Gym × Slime GPU dogfood.

## Status snapshot

**Repo:** `main` (edge-case recipes + `dg ls` / `dg stats`)  
**GPU:** Keep the live RunPod H100 if still up — use it for edge recipes below  
**Local path:** `~/Desktop/projects/daytona-training-gym-spec`  
**Pod path:** `/root/daytona-training-gym-spec`, Slime `/root/slime`, Megatron `/root/Megatron-LM`

### Proven on GPU

| Check | Result |
|---|---|
| Slime train + Daytona custom generate/RM | Pass (0.5B → 3B) |
| Sandbox provision → seed → bootstrap → finalize | Pass |
| Stress 40 rollouts (`4×2×5`) all `reward=1.0` | Pass |
| Model-emitted tools + coding `reward=1.0` (3B) | Pass (`Ah8AjLWwnxfDK7Qe`) |
| `dg ls` / `dg stats` | Pass |
| Customer edge recipes (stall / OOM / multifile) | **In progress — run on H100** |

### Important pod state

- Models: `Qwen2.5-3B-Instruct` (+ `_torch_dist`); older 0.5B/1.5B may still be on disk
- API key via `DAYTONA_API_KEY_FILE` only in Ray `runtime_env`
- **Rotate keys** that ever leaked in chat / old `ray job list`

---

## Next on the live H100 (priority)

Run edge recipes one at a time; capture failures in `FRICTION.md`:

```bash
cd /root/daytona-training-gym-spec && git pull && pip install -e .

bash examples/coding_dogfood/run_edge_case.sh tool_stall
dg stats runs/edge_tool_stall.jsonl

bash examples/coding_dogfood/run_edge_case.sh rollout_budget
bash examples/coding_dogfood/run_edge_case.sh concurrency_storm
bash examples/coding_dogfood/run_edge_case.sh mem_pressure
bash examples/coding_dogfood/run_edge_case.sh hard_prompts
bash examples/coding_dogfood/run_edge_case.sh wrong_bootstrap
```

Then Harbor spike (P1b) on CPU when edge findings are logged.

---

## Backlog

### P1 — DX

- [ ] Partner dogfood blurb → `examples/coding_dogfood/README.md` + preflight
- [ ] Optional: parameterize remaining pod sed hacks away from MODEL_SCRIPT env

### P1b — Harbor second adapter

- [ ] Spike Harbor Daytona config surface
- [ ] Sketch `adapters/harbor/` + fake caller CPU test

### P2 — Optional

- [ ] 7B when a partner needs it
- [ ] Soft-fail mixture recipes with curated abort rates

---

## Key files

| File | Why |
|---|---|
| `examples/coding_dogfood/run_edge_case.sh` | Stall / OOM / concurrency / hard prompts |
| `examples/coding_dogfood/run_on_slime_pod.sh` | Base GPU launcher |
| `examples/coding_dogfood/GPU_RUN.md` | Pass criteria + edge checklist |
| `examples/coding_dogfood/FRICTION.md` | War stories |
| `daytona_gym/cli.py` | `dg` / `dg ls` / `dg stats` |

---

*Next: run edge recipes on H100 → log FRICTION → Harbor spike.*
