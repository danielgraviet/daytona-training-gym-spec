# Customer dogfood notes — RunPod A100 + Qwen “LeetCode agent”

Date: 2026-09-24  
Constraint: **no edits under `daytona_gym/`** — customer path only.  
Pod: `RUNPOD_POD_ID` existing slime image (`slimerl/slime:latest`, RUNNING).  
Model: `Qwen25_3B()` / Qwen2.5-3B-Instruct on 1× GPU.

## What we ran

### Phase 0 — Preflight

- Env keys present (`DAYTONA_API_KEY`, `RUNPOD_API_KEY`, `RUNPOD_POD_ID`, SSH identity).
- Dry-run: `python examples/gym_sdk/remote_from_laptop.py` → worker `…@ssh.runpod.io`, plan OK.
- API: image `slimerl/slime:latest`, proxy SSH user present.

### Phase 1 — Proven coding dogfood

```bash
python examples/gym_sdk/remote_from_laptop.py --launch --detach
```

- Run id: `run_369be3df4edb46c58cadd6d3695dbb52`
- Laptop `TrainingRun.result()` → **Training complete** (~6.5 min wall, mostly Megatron init).
- Telemetry pulled via `PtyShell` (not `dg dash --remote` — see holes).
- `dg stats`: **n=1, status=completed, reward=1.0**, tools `run_tests×2 write_file×1`.

**Pass criteria: met** for “Gym works on my A100 + Daytona credits.”

### Phase 2 — LeetCode-flavored stretch (prompt-only)

Artifacts (customer tree, not SDK):

- [`leetcode_easy.jsonl`](leetcode_easy.jsonl) — Two Sum / Valid Parentheses prose + **same** coding-seed tool contract.
- [`launch_leetcode_stretch.py`](launch_leetcode_stretch.py) — thin TrainConfig launcher.

Had to **manually upload** JSONL to the pod (chunked base64 over PtyShell) because the file is not on the git remote the worker `git pull`s.

```bash
python examples/customer_dogfood/launch_leetcode_stretch.py --launch --detach
```

- Run id: `run_088c62a79fd4469b9af1ff583587fb53`
- **Training complete**, reward **1.0**, still `write_file` → `broken.py` `add(a,b)`.
- Generation preview: model fixed `add`, **ignored LeetCode framing**.

**Primary product hole confirmed:** dataset prompt cannot replace sandbox seed (`_coding_seed.py` basic profile).

---

## Holes found (customer-facing)

1. **No LeetCode / custom task pack** — seed is hardcoded `broken.py` / `test_broken.py` (or multifile). Prompt-only “LeetCode” still trains on add(a,b).
2. **Customer data not on pod** — uncommitted `examples/customer_dogfood/*` must be uploaded by hand; `git pull` on worker won’t see it. No first-class dataset sync.
3. **`remote_from_laptop.py` hardcoded** to `Qwen25_3B` + `coding_one.jsonl` — customer forks or uses a side script (we did).
4. **`dg dash --remote` fails in Cursor agent shell** — RunPod proxy needs a real `/dev/tty`; error tells you to use Mac Terminal. Agent dogfood cannot rely on remote dash pull.
5. **Detach dash still Cloudflare on this pod** — returned `trycloudflare.com` URL (pod-side `open()` / older default). Laptop-local preference requires `--remote` from a real Terminal or accepting tunnel.
6. **`dg run status <id>` on laptop** sees nothing until JSONL is copied locally — progress/telemetry live on the GPU box; no automatic laptop mirror after detach.
7. **Raw `ssh host 'cmd'` does not work** on RunPod proxy — must use PTY shell transport (SDK does; naive customer scripts may not).
8. **Secrets in remote command line** — launch path exports `DAYTONA_API_KEY` / `HF_TOKEN` into the typed shell command (visible in session logs). Rotate if logs were shared.
9. **Long silent Megatron init** — CLI stage spam “Initializing Megatron…” for ~5+ minutes; dash helps only if reachable.
10. **No post-train LeetCode eval / serve** — only train rollouts + `dg` inspect.
11. **`HuggingFaceDataset.materialize()`** does not produce seeded sandbox tasks — useless alone for this agent loop.
12. **7B / Coder on 1× A100** — not attempted; presets exist but unverified (separate risk).

## What worked well

- Laptop → RunPod proxy → detach → `Training complete` banner (Modal-shaped).
- Daytona sandbox tool loop + reward=1.0 on stock coding seed (Phase 1 and 2).
- Qwen2.5-3B tool use (`write_file` + `run_tests`) on this box.
- Preflight / slime image assumption held for an existing customer pod.

## Recommended product follow-ups (for later SDK work — not done here)

- Dataset-driven sandbox seeds (or Harbor tasks) for real coding/LC problems.
- First-class `dg dash --remote` that works without a local TTY, or auto-sync `runs/` after detach.
- Example CLI flags for model/dataset (stop hardcoding `remote_from_laptop.py`).
- Default localhost dash on workers; document Terminal-only `--remote`.
- Avoid putting raw API keys in remote shell command lines.
