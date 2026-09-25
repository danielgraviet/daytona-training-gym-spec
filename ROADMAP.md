# Roadmap — next build phase

Status: **active plan** (written 2026-09-25 after a repo-wide review; updated as
work lands). Supersedes the "Next" section of `TODO.md` and the "Milestone 0
only" guidance in `AGENTS.md`.

How to use this file: each task is a checkbox with a **Done when** line. Tick
it only when the done-when is met, and leave a short note (what shipped, where,
how it was verified). ⏳ marks work that is partly done.

## At a glance

| Phase | State | One-line summary |
| --- | --- | --- |
| 0 Hygiene | ✅ done | Tests green, secrets off command lines, docs point here |
| 1 Flagship analytics | ✅ done (1.2 partial) | Per-step "where time went" + $ — verified live on A100 |
| 2 Durable telemetry | ✅ built, ⏳ live verification | `dg ingest` + worker shipper + `worker_lost`; needs a hosted deploy + real pod kill |
| 3 BYO portability | not started | Worker image, outbound agent, second provider |
| 4 Real task packs | not started | Dataset-driven seeds, reward callable, repo-scale pack |
| 5 Analytics-only path | not started | Slime users anywhere get the dashboard via adapter + ingest token |

### Next up

1. **Decide where `dg ingest` runs** (see Phase 2 → open decision) and do the
   Phase 2 live verification on a RunPod A100.
2. Finish **1.2** (Slime `perf/*` ingest) — the log format is now known from the
   live run, so this is a small parser, not a spike.
3. Start **Phase 3**.

## Why this phase

The sandbox rollout loop is proven (H100/A100 dogfood, edge suite, 48/48
concurrency) and the Modal-shaped SDK works laptop → RunPod. What was **not**
yet real are the two things we sell against Modal:

1. **Analytics** — "where did my rollout / GPU time go?" (now shipped, Phase 1).
2. **BYO GPU** — today means "RunPod pod running `slimerl/slime`". Portability
   is claimed, not demonstrated (Phase 3).

And the stickiness thesis ("run history is the sticky layer",
`PRODUCT_DECISIONS.md` §13) needs run history that outlives the pod (Phase 2).

### Unifying pitch

> **"Here is how many GPU-dollars you spent waiting on environments."**

The analytics exposes idle-GPU waste; BYO lets the customer act on it (cheaper
GPUs, fewer GPUs, faster sandboxes). Modal has little incentive to surface idle
time on GPUs it bills for.

---

## Phase 0 — Hygiene ✅

- [x] **0.1 Fix failing tests.** Done when `pytest` is fully green.
  - Note: `wait()` had moved to Rich `WaitProgress`; tests updated, dead
    `TrainingRun._emit_stage` removed. Also fixed a real bug: the breadcrumb
    printed the *new* message under the old phase, and Rich swallowed
    `[phase]` as markup.
- [x] **0.2 Key hygiene.** Done when the plaintext key is gone and the old key is revoked.
  - Note: removed from `scratch.txt`; user rotated the Daytona key 2026-09-25.
- [x] **0.3 Secrets off command lines.** Done when a contract test proves no
  secret value appears in payloads, argv, or captured shell output.
  - Note: job payload carries key *names* only (`forward_env_keys`); env file is
    0600, sourced then deleted; PTY echo is disabled while base64 chunks are
    typed. `tests/contracts/test_secret_forwarding.py`.
- [x] **0.4 Docs match code.**
  - Note: `AGENTS.md`, `TODO.md`, `COMPETITIVE_MODAL_GYM.md`, `README.md` point here.

---

## Phase 1 — Flagship analytics: time-aligned, step-level, in dollars ✅

Goal: open a run and immediately see, per training step, whether the GPU was
waiting on environments, which rollouts caused it, and what that cost.

- [x] **1.1 Populate correlation IDs.** Done when every span of a live run has
  non-empty `worker_id`, `training_step`, `rollout_batch_id`, enforced by a
  contract test through the Slime adapter.
  - Note: `worker_id` = `DAYTONA_WORKER_ID` → `runpod:<pod>` → hostname.
    Slime's custom-generate hook does not expose the step, so it is **derived**
    (`telemetry/batches.py`): a new step starts after all rollouts drained and
    stayed idle ≥ 1s (`DAYTONA_BATCH_GAP_SECONDS`). Explicit ids win; spans
    record `training_step_source`. Provision span now carries `sandbox_id`;
    `sandbox.seed` counts as sandbox time. Analysis groups by
    `(rollout_id, step)` in case a trainer reuses sample indices.
    `tests/contracts/test_correlation_ids.py`.
- [ ] ⏳ **1.2 Trainer-side timing ingest.** Done when the dashboard shows a
  per-step bar `rollout | train | weight_sync | other` from trainer-reported numbers.
  - Shipped so far: train phase is *derived* from the gap between steps.
  - Remaining: parse Slime's `perf/*` dict lines from the Ray job output that
    `execute_plan` already streams (seen live: `perf/step_time`,
    `perf/wait_time_ratio`, `perf/actor_train_tflops`, …), emit them as
    `train.*` metrics keyed by step, and show "Slime-reported" next to derived
    values. Step index = order of appearance unless the line carries one.
- [x] **1.3 Time-aligned GPU view.** Done when GPU util and rollout spans share a wall-clock axis.
  - Note: `WhereTimeWent.svelte` timeline — rollouts, inference segments,
    GPU-idle windows, GPU util on one axis; GPU sampler every 5s.
- [x] **1.4 Derived metrics with definitions.** Done when each metric is unit
  tested on synthetic spans and its definition is shown in the UI.
  - Note: `telemetry/analysis.py` — env wait (no inference in flight), straggler
    tax, per-step rollout bound, three-way run **bottleneck**
    (environment / inference / trainer with shares), failed-rollout waste,
    provision / tool percentiles. `tests/contracts/test_analysis.py`.
- [x] **1.5 Dollars.** Done when `$` idle cost shows when a rate is set and hides otherwise.
  - Note: `TrainConfig(gpu_cost_per_hour=...)` → `run.meta` line in telemetry;
    `dg stats --gpu-cost-per-hour/--num-gpus` override.
- [x] **Phase 1 exit — verified live.** A user can answer "which step was slow,
  why, which rollout, what it cost" from the dashboard alone.
  - Note: A100 80GB, `slimerl/slime:latest`, run `run_b17b5158…` (2026-09-25):
    4 steps × 8 rollouts detected correctly; 32/32 reward 1.0; env wait 46% of
    rollout phase; derived train phase 98s vs ~20s of rollouts → **trainer-bound
    (~83%)**, matching Slime's own `perf/wait_time_ratio≈0.96`.

Phase 1 follow-ups found during the live run (all fixed):

- [x] On-pod `launch(open=True)` returned an unreachable `127.0.0.1` URL → tunnels by default on RunPod/SSH.
- [x] `dg dash` on the pod tried to SSH-pull from itself (+ `text[-240]` crash) → serves local runs.
- [x] Dashboard said "running" after success; opened only after training → progress lifecycle written, dash opens first; legacy runs show `stale`.
- [x] Raw Cloudflare 530 HTML after Ctrl+C → offline banner, keeps last data, polling backs off.

---

## Phase 2 — Durable, off-box telemetry (the sticky layer) ✅ built / ⏳ live check

Goal: run history outlives the pod; laptop sees live status after detach; no
Cloudflare dependency.

**Design (as built).** Instead of a second exporter inside the rollout process,
a **shipper thread on the launch host tails the files the run already writes**
(`runs/<id>.jsonl`, `runs/<id>.progress.json`) and pushes them to
**`dg ingest`**, which stores the same `runs/` layout and serves the existing
dashboard. This covers every producer (rollout spans, GPU sampler, run meta,
progress) with zero changes to the rollout hot path.

```text
GPU worker (any provider)                       ingest host (dg ingest)
  rollout actor ─► runs/<id>.jsonl ─┐           <data>/<id>.jsonl
  GPU sampler  ─►       "           ├─ shipper ─► <data>/<id>.progress.json ─► dashboard + /api
  progress     ─► <id>.progress.json┘  (HTTPS)  <data>/<id>.ingest.json (offset, last_seen)
```

Configure on the laptop or pod (forwarded to workers via the 0600 env file):

```bash
export DAYTONA_GYM_INGEST_URL=https://gym.example.com
export DAYTONA_GYM_INGEST_TOKEN=...   # ≥16 chars; same token opens the dashboard
```

- [x] **2.1 Exporter boundary.** Done when remote shipping needs no change to the rollout path.
  - Note: existing `TelemetryExporter` / `AsyncJsonlExporter` unchanged as the
    local sink; shipping is a separate reader (`daytona_gym/ingest/shipper.py`).
- [x] **2.2 Remote push with retries.** Done when pushes are batched,
  idempotent, back off on failure, and never block or crash the launch.
  - Note: chunks tagged with **source** byte offsets (`X-DG-Offset/Next`); the
    server appends only on a matching offset, else 409 + its offset → shipper
    resyncs (safe across retries and worker restarts). Truncation/rotation bumps
    `X-DG-Generation`. Oversized lines are skipped (counted), not stalled.
    Backoff ≤60s; data stays on local disk while the ingest host is down.
    Heartbeat every 15s.
- [ ] **2.3 OTLP export option.** Done when spans can also go to a customer's
  OpenTelemetry collector (`PRODUCT_DECISIONS.md` §8).
  - Not started — deferred within the phase; the shipper is the place to add an
    OTLP/HTTP JSON sink. No dependency needed.
- [x] **2.4 `TrainingRun.wait()` reads the ingest API.** Done when a detached
  laptop session gets live status without SSH/PTY sync.
  - Note: with ingest configured, `run.open()` / the laptop detach path return
    `<ingest>/run/<id>` (no local server, no tunnel); `wait()` polls
    `/api/runs/<id>/live` with the bearer token (sent only to the ingest host).
    `worker_lost` ends `wait()` as failed.
- [x] **2.5 Dashboard served from the ingest store.** Done when the same SPA
  works off ingest data and Cloudflare is only a fallback.
  - Note: `dg ingest` routes GETs to the dashboard; tunnel is used only when no
    ingest is configured. Browser login at `/login` sets an HttpOnly,
    SameSite=Strict cookie (`Secure` behind HTTPS).
- [x] **2.6 Redaction before export.** Done when a contract test proves secret
  values never reach the ingest host.
  - Note: shipper scrubs values of secret-looking env vars in its process plus
    `dtn_…`, `hf_…`, `Bearer …` patterns from telemetry and progress before
    sending (on top of key-based attribute redaction upstream).
- [x] **Security hardening that came with a public dashboard.**
  - Note: token required for reads and writes (server refuses to start without
    one unless `--no-auth`); run ids validated (no traversal); body ≤8 MB,
    must be JSON-object lines; `_resolve_run` can no longer escape the runs dir.
- [x] **Phase 2 exit — verified locally.** Kill the worker mid-run; run and
  telemetry up to that point remain browsable; status shows `worker_lost`.
  - Note: automated (`tests/contracts/test_ingest.py`, 12 tests) and a live
    process check: `dg ingest` + a fake worker that shipped 6 rollouts then
    `kill -9`'d itself → all 6 rollouts browsable, planted API key absent from
    the store, unauthenticated reads 401, status `worker_lost` after 90s silence.
- [ ] **Open decision: where `dg ingest` runs in production.** Done when there
  is one documented deployment with HTTPS.
  - Options: (a) small VM / Fly / Render with a persistent volume (simplest);
    (b) a long-lived **Daytona sandbox** with a preview URL (dogfoods Daytona,
    needs auto-stop disabled + volume); (c) Daytona-hosted multi-tenant service
    later (needs per-org tokens).
- [ ] **Phase 2 exit — verified live on RunPod.** Done when:
  - [ ] `dg ingest` deployed behind HTTPS with a real token
  - [ ] `python main.py` on the A100 prints the ingest URL (not trycloudflare / 127.0.0.1)
  - [ ] dashboard updates live during training from the ingest host
  - [ ] terminate the pod mid-run → run stays browsable, shows `worker_lost` within ~90s
  - [ ] laptop `launch(worker=runpod_worker(), detach=True)` + `run.wait()` works with no PTY sync

---

## Phase 3 — Real BYO portability

Goal: "move off Modal's GPUs to anything" is demonstrated, not claimed.

- [ ] **3.1 Worker image contract.** Done when a published image (Slime +
  Megatron + SGLang + `daytona_gym`) runs the gym with documented paths and no
  `/root/slime` assumptions or git clone on the worker
  (`LocalSlimeCompute` defaults, `remote_job.py`).
- [ ] **3.2 Outbound worker agent** (`daytona_gym/workers/agent.py`). Done when
  `docker run <image> daytona-gym worker --token …` dials out, receives a
  launch payload, reports lifecycle + GPU inventory, and supports safe stop —
  no inbound SSH, no PTY proxy hacks (`SshWorker` stays as an escape hatch).
  - Context: the ingest host from Phase 2 is the natural control plane
    endpoint for the agent to poll.
- [ ] **3.3 Dataset/config shipped with the launch payload.** Done when a local
  JSONL not in git trains on the worker (customer note: `git pull` can't see it).
- [ ] **3.4 Dogfood on a second provider** (Lambda or a bare SSH/on-prem box).
  Done when the same `TrainConfig` trains there with zero code changes;
  results recorded in `FRICTION.md`.

**Phase 3 exit:** same script trains on RunPod and one non-RunPod provider;
only the worker token/target differs.

---

## Phase 4 — Real task packs (why Daytona sandboxes matter)

Goal: move past the hardcoded `add(a,b)` seed.

- [ ] **4.1 Dataset-driven seeds by default.** Done when each row's
  `seed_files` / repo ref + `run_tests` command drives the sandbox and the silent
  fallback to the `basic` profile is gone (`seed_files_from_label` exists).
- [ ] **4.2 Python reward callable** on `TrainConfig`/recipe (not just an
  `rm_path` string). Reward semantics stay user-owned.
- [ ] **4.3 Repo-scale task pack** (small real repo, multi-file fix, real test
  suite) using a **Daytona snapshot**; dashboard shows provision p95 with vs
  without snapshot.
- [ ] **4.4 Rename `CodingRecipe` presets** away from implying the add-bug task.

**Phase 4 exit:** the LeetCode customer dogfood from
`examples/customer_dogfood/NOTES.md` trains on its own tasks.

---

## Phase 5 — Analytics-only adoption path

- [ ] Teams already running Slime (anywhere, including on Modal) get the
  dashboard by pointing `--custom-generate-function-path` at our adapter plus
  `DAYTONA_GYM_INGEST_URL/TOKEN` — no launcher, no worker. One page, copy-paste
  snippet. Done when a Slime run launched outside `TrainConfig` shows up in
  `dg ingest` with the full "where time went" view.
  - Context: Phase 2 makes this mostly documentation + a standalone shipper
    entrypoint (`python -m daytona_gym.ingest.shipper runs/<id>.jsonl`).

---

## Deferred (explicitly not this phase)

- Harbor backend (stub stays; revisit after Phases 1–3 since it reuses their analytics + ingest)
- More model presets / recipe catalog
- Agent skills bundle polish
- PyPI release
- Daytona-managed GPUs

## Order and parallelism

```text
Phase 0 ──► Phase 1 ──► Phase 2 (built) ──► live verify ──► Phase 5
                   └──► 1.2 (small, any time)
Phase 3 can start now (agent reports to the Phase 2 ingest host)
Phase 4 after Phase 3.3 (task data must reach the worker)
```

## Constraints (unchanged, from AGENTS.md)

- No new RL trainer; no Slime fork. Slime/Harbor types stay under `adapters/`.
- Async everywhere; cleanup on success/exception/cancel/timeout.
- Telemetry must not materially block rollouts.
- Live Daytona e2e is first-class; `FakeAsyncDaytona` mirrors observed SDK behavior only.
- Never log secrets; redact env; cap captured output.

## Log

| Date | Commit | What |
| --- | --- | --- |
| 2026-09-25 | `ea892aa` | Phase 0 + Phase 1 analytics, secret forwarding hardening |
| 2026-09-25 | `bd77aeb` | On-pod launches tunnel the dashboard by default |
| 2026-09-25 | `01c7ec0` | `dg dash` on the pod serves local runs |
| 2026-09-25 | `75091bf` | Progress lifecycle for on-box launches; dash opens before training |
| 2026-09-25 | `b310806` | Three-way run bottleneck (environment / inference / trainer) |
| 2026-09-25 | `49e2870` | Offline banner instead of raw Cloudflare 530 HTML |
| 2026-09-25 | see `git log -- daytona_gym/ingest` | Phase 2: `dg ingest`, worker shipper, `worker_lost`, auth + hardening |
