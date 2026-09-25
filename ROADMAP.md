# Roadmap — next build phase

Status: **active plan** (written 2026-09-25 after a repo-wide review).
Supersedes the "Next" section of `TODO.md` and the "Milestone 0 only" guidance in `AGENTS.md`.

## Progress

| Item | Status | Notes |
| --- | --- | --- |
| 0.1 failing tests | ✅ done | Also fixed a real `WaitProgress` bug: breadcrumb printed the *new* message under the old phase, and Rich swallowed `[phase]` as markup |
| 0.2 key hygiene | ⚠️ partly done | Key removed from `scratch.txt`; **rotation is still a manual step** (Daytona console) |
| 0.3 secrets off command lines | ✅ done | Payload carries key *names* only; env file 0600, sourced then deleted; PTY echo disabled while typing; `test_secret_forwarding.py` |
| 0.4 docs | ✅ done | `AGENTS.md`, `TODO.md`, `COMPETITIVE_MODAL_GYM.md` point here |
| 1.1 correlation ids | ✅ done (needs live check) | `worker_id`, `training_step`, `rollout_batch_id` on every span; `sandbox_id` on provision; step is *derived* (`telemetry/batches.py`) unless trainer passes one |
| 1.2 trainer-side timing | ⏳ partial | Train phase derived from rollout gaps. Slime timer ingest still needs a spike against a real Slime install |
| 1.3 time-aligned GPU view | ✅ done | Timeline: rollouts + inference + GPU-idle windows + GPU util on one axis; GPU chart on wall-clock |
| 1.4 derived metrics | ✅ done | `telemetry/analysis.py`: env wait, straggler tax, bound, failed waste, provision/tool percentiles, definitions |
| 1.5 dollars | ✅ done | `TrainConfig(gpu_cost_per_hour=...)` → `run.meta` in telemetry; `dg stats --gpu-cost-per-hour` override |
| Phase 1 exit | ⏳ pending | Verify on a live multi-step BYO run (derived step ids + GPU sampler at 5s) |
| Phases 2–5 | not started | |

## Why this phase

The sandbox rollout loop is proven (H100/A100 dogfood, edge suite, 48/48
concurrency) and the Modal-shaped SDK works laptop → RunPod. What is **not**
yet real are the two things we sell against Modal:

1. **Analytics** — "where did my rollout / GPU time go?" Today we sum
   per-rollout spans into 5 buckets. Modal already shows a per-step training
   wall-time breakdown, so on this axis we currently show *less* than Modal.
2. **BYO GPU** — today means "RunPod pod running `slimerl/slime`". Portability
   is claimed, not demonstrated.

And the stickiness thesis ("run history is the sticky layer",
`PRODUCT_DECISIONS.md` §13) is contradicted by telemetry living as JSONL on a
GPU box that disappears with the pod.

### Unifying pitch for this phase

> **"Here is how many GPU-dollars you spent waiting on environments."**

This ties both differentiators together: the analytics exposes idle-GPU waste,
BYO lets the customer act on it (cheaper GPUs, fewer GPUs, faster sandboxes).
Modal has little incentive to surface idle time on GPUs it bills for.

---

## Phase 0 — Hygiene (≈1 day, do first)

| # | Task | Where | Done when |
| --- | --- | --- | --- |
| 0.1 | Fix 3 failing tests: `wait()` now renders via Rich `WaitProgress`, tests still assert old `▶ [` lines. Update tests to assert on `WaitProgress` output (or a `quiet`/plain mode), and delete dead `TrainingRun._emit_stage`. | `tests/contracts/test_training_run_wait.py`, `daytona_gym/gym/run.py:302` | `pytest` fully green |
| 0.2 | Rotate the Daytona API key that sat in plaintext in `scratch.txt`; delete the key from that file. | local | Old key revoked |
| 0.3 | Stop putting secrets on remote command lines. Forward `DAYTONA_API_KEY` / `HF_TOKEN` via a `0600` env file written over the shell channel (or stdin), then `source` + `rm`. Never echo values. | `daytona_gym/gym/worker.py` (`_export_forward_env_cmd`, `_install_forward_env_via_shell`) | Contract test asserts no secret value appears in any command string / captured output |
| 0.4 | Rewrite `AGENTS.md` "First implementation target" to point at this roadmap; refresh `TODO.md` and the status tables in `COMPETITIVE_MODAL_GYM.md` (TrainingRun, SshWorker, detach are shipped). | docs | Docs match code |

---

## Phase 1 — Flagship analytics: time-aligned, step-level, in dollars

Goal: open a run and immediately see, per training step, whether the GPU was
waiting on environments, which rollouts caused it, and what that cost.

### 1.1 Populate correlation IDs (prerequisite — currently empty)

Real runs emit `training_step=""`, `rollout_batch_id=""`, `worker_id=""`, and
`sandbox_id=""` on `sandbox.provision` (see `runs/run_369be…jsonl`). Nothing
step-level is possible until these are filled.

- `worker_id`: set on the worker at launch (hostname + provider id), pass via env → `args`.
- `training_step` / `rollout_batch_id`: spike how to obtain from Slime inside
  `custom_generate` (e.g. rollout id / step on `args` or sample metadata). If
  Slime does not expose it, derive a batch id from the rollout-manager call
  boundary and document it as derived.
- `sandbox_id`: set on the provision span once the sandbox is created (span attribute update before close).
- Make `sandbox.seed` categorize as `sandbox`, not `other` (`telemetry/store.py:354`).

**Done when:** a live run's JSONL has non-empty `worker_id`, `training_step`,
`rollout_batch_id` on every span; contract test enforces it through the Slime adapter.

### 1.2 Trainer-side timing ingest

- Spike: where Slime exposes per-step timings (rollout generation, logprobs,
  train, weight sync, checkpoint) — its timer logs / wandb / tensorboard
  outputs. Prefer reading an existing sink over patching Slime (no Slime fork).
- Emit them into our store as `train.*` spans keyed by `training_step`.
- Fallback if no clean hook: derive "rollout phase" per step from first/last
  rollout span in the batch and treat the gap to the next batch as "train phase".

**Done when:** dashboard shows a per-step bar: `rollout | train | weight_sync | other`.

### 1.3 Time-aligned GPU view

- Plot GPU util/memory on **wall-clock time**, not sample index
  (`dashboard_frontend/src/pages/RunDetail.svelte:124`, `dashboard_data.run_charts`).
- Overlay rollout spans (swimlane/Gantt per rollout) on the same axis.

### 1.4 Derived metrics (from `OBSERVABILITY.md`, now implemented)

Put these in a pure module (`daytona_gym/telemetry/analysis.py`) with unit tests
on synthetic span sets. Each metric ships with its definition shown in the UI.

- **Critical-path decomposition** per step: replace the naive sum in
  `wall_time_decomposition` for run-level views; attribute the step's rollout
  phase to the rollout(s) that ended last.
- **Straggler tax** per step: `last_rollout_end − p50_rollout_end` (baseline configurable).
- **Env-bound vs inference-bound** verdict per step and per run.
- **GPU rollout-starvation estimate**: GPU util below threshold while rollouts
  are in flight and no train span is active. Labelled "estimate".
- **Failed-rollout waste**: GPU generation seconds + sandbox seconds on rollouts that ended failed/aborted.
- **Sandbox provision p50/p95/p99**, tool p95 by tool name.

### 1.5 Dollars

- `TrainConfig(..., gpu_cost_per_hour=...)` (and per-worker override). Optional;
  hide `$` columns if unset.
- Headline tile: **"GPU $ idle waiting on environments"** + % of run cost.

**Phase 1 exit:** on a real BYO run, a user can answer "which step was slow,
why, which rollout, and what it cost" from the dashboard alone, without `dg` or grep.

---

## Phase 2 — Durable, off-box telemetry (the sticky layer)

Goal: run history outlives the pod; laptop sees live status after detach; no
Cloudflare dependency.

- 2.1 Define an exporter interface in `telemetry/` (keep existing async JSONL
  exporter as the default local sink). Must stay non-blocking with bounded buffers.
- 2.2 Add a remote exporter: batched HTTPS push to an ingest endpoint (start
  with a minimal self-hostable service + object storage; Daytona-hosted later).
  Retries with backoff; drops + counts on overflow, never blocks rollouts.
- 2.3 OTLP export option for spans (honours `PRODUCT_DECISIONS.md` §8) so
  customers can also send to their own backend.
- 2.4 `TrainingRun.wait()` / `dg run status` read from the ingest API instead of tunnelling to the pod.
- 2.5 Dashboard serves from the ingest store; Cloudflare tunnel becomes a fallback only.
- 2.6 Redaction applies before export (env vars, secrets, capped stdout/stderr) — contract test.

**Phase 2 exit:** kill the pod mid-run; the run and all telemetry up to that
point remain browsable, and status shows `worker_lost`.

---

## Phase 3 — Real BYO portability

Goal: "move off Modal's GPUs to anything" is demonstrated, not claimed.

- 3.1 **Worker image contract**: publish a container image (Slime + Megatron +
  SGLang + `daytona_gym`) with documented paths; stop assuming `/root/slime`
  and git-cloning this repo on the worker (`LocalSlimeCompute` defaults, `remote_job.py`).
- 3.2 **Outbound worker agent** (`daytona_gym/workers/agent.py`): `docker run
  <image> daytona-gym worker --token …` dials out to the control plane /
  ingest, receives a launch payload, reports lifecycle + GPU inventory, supports safe stop.
  No inbound SSH, no PTY proxy hacks. `SshWorker` stays as an escape hatch.
- 3.3 Dataset/config shipped with the launch payload (fixes customer note:
  local JSONL not visible to the worker's `git pull`).
- 3.4 **Dogfood on a second provider** (Lambda or a bare SSH/on-prem box) with
  the same `TrainConfig` and zero code changes. Record results in `FRICTION.md`.

**Phase 3 exit:** same script trains on RunPod and one non-RunPod provider;
only the worker token/target differs.

---

## Phase 4 — Real task packs (why Daytona sandboxes matter)

Goal: move past the hardcoded `add(a,b)` seed.

- 4.1 Dataset-driven seeds as the default path: each row carries
  `seed_files` / repo ref + `run_tests` command (`seed_files_from_label` exists;
  make it first-class and documented, remove the silent fallback to the `basic` profile).
- 4.2 `TrainConfig`/recipe accepts a **Python reward callable** (not just an
  `rm_path` string); still user-owned semantics.
- 4.3 One repo-scale task pack (small real repo, multi-file fix, real test
  suite) using a **Daytona snapshot** for fast provisioning; show provision
  p95 with vs without snapshot in the dashboard.
- 4.4 Rename `CodingRecipe` presets away from implying the add-bug task.

**Phase 4 exit:** the LeetCode customer dogfood from `examples/customer_dogfood/NOTES.md` trains on its own tasks.

---

## Phase 5 — Analytics-only adoption path

Let teams already running Slime (anywhere, including on Modal) get the
dashboard by pointing `--custom-generate-function-path` at our adapter plus an
ingest token — no launcher, no worker. Documented in one page with a
copy-paste snippet. Lowest-friction way to prove the differentiator.

---

## Deferred (explicitly not this phase)

- Harbor backend (stub stays; revisit after Phases 1–3 since it reuses their analytics + ingest)
- More model presets / recipe catalog
- Agent skills bundle polish
- PyPI release
- Daytona-managed GPUs

## Order and parallelism

```text
Phase 0 ──► Phase 1.1 ──► 1.2 / 1.3 / 1.4 (parallel) ──► 1.5
                 └──► Phase 2 (can start after 1.1; schema must be stable)
Phase 3 can start in parallel with Phase 2 (agent reports to ingest)
Phase 4 after Phase 1 (so the demo shows analytics on real tasks)
Phase 5 after Phase 2
```

## Constraints (unchanged, from AGENTS.md)

- No new RL trainer; no Slime fork. Slime/Harbor types stay under `adapters/`.
- Async everywhere; cleanup on success/exception/cancel/timeout.
- Telemetry must not materially block rollouts.
- Live Daytona e2e is first-class; `FakeAsyncDaytona` mirrors observed SDK behavior only.
- Never log secrets; redact env; cap captured output.
