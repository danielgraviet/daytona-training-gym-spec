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
| 1 Flagship analytics | ✅ done (1.2 live check pending) | Per-step "where time went" + $ — verified live on A100 |
| 2 Durable telemetry | ✅ done, verified live | `dg ingest` on a Daytona sandbox; A100 run shipped live; pod stop → `worker_lost` (2.3 OTLP deferred) |
| 3 BYO portability | ⏳ in progress | 3.3 done; 3.1 image published (GHCR + Docker Hub); 3.4 home 3090 trains all steps, fails after — resume Monday |
| 4 Real task packs | not started | Dataset-driven seeds, reward callable, repo-scale pack |
| 5 Analytics-only path | not started | Slime users anywhere get the dashboard via adapter + ingest token |

### Next up

1. ~~Confirm 1.2 on A100~~ — confirmed (`run_ecc7…`: Slime split live; 3090 runs too).
2. **Monday: finish 3.4 on the home 3090** (checklist under 3.4) — all 4
   steps already train; the failure is after training.
3. Then **3.2 outbound agent**, and the ingest watchdog (sandbox was found stopped).

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
- [x] **1.2 Trainer-side timing ingest.** Done when the step view splits
  trainer time using trainer-reported numbers (⏳ live A100 confirmation pending).
  - Note: `telemetry/slime_perf.py` parses Slime's per-step `perf/*` dict from
    the Ray job output the launcher already streams (`ast.literal_eval`, never
    eval; step from the line or order of appearance; duplicate lines deduped)
    and appends `slime.perf.*` gauges to the run JSONL. Per step:
    train = `step_time × (1 − wait_time_ratio)`, overhead = Slime wait − the
    Daytona-observed rollout phase (engine offload/onload, weight sync). The
    run bottleneck becomes environment / inference / trainer / **overhead**
    when Slime numbers exist, else falls back to the derived gap. On a run
    shaped like `run_b959…` this reads **overhead ≈ 88%, trainer ≈ 4%** — the
    "trainer-bound" verdict was really engine-swap/sync-bound.
    `tests/contracts/test_slime_perf.py`.
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
- [x] **Hosting: long-lived Daytona sandbox (dogfood) — `dg ingest deploy --daytona`.**
  Done when one command gives an HTTPS ingest URL + token and is safe to re-run.
  - Note: idempotent — finds the sandbox by label, starts it if stopped,
    re-asserts auto-stop 0, `pip install`s the gym from `main`, starts
    `dg ingest` if not healthy, checks `/healthz` through the preview proxy,
    writes/prints `DAYTONA_GYM_INGEST_URL/TOKEN`. VM/Fly/Render stays the
    documented fallback (`dg ingest --host 0.0.0.0` behind TLS).
  - Live probe 2026-09-25 (sandbox created, tested, deleted):
    | Check | Result |
    | --- | --- |
    | Our 401 / JSON passes through the preview proxy unchanged | ✅ |
    | `Authorization: Bearer` reaches `dg ingest` | ✅ |
    | 3.3 MB shipped in 1 MB chunks | ✅ 2.2s, 3000/3000 rollouts |
    | Browser login → cookie gets `Secure` (proxy sets `X-Forwarded-Proto: https`) | ✅ |
    | `auto_stop_interval=0`, `auto_delete_interval=-1`, `auto_pause_interval` already 0 | ✅ |
    | Stop → start: preview URL unchanged, data on disk intact | ✅ |
    | Stop → start: `dg ingest` process survives | ❌ proxy 502 until re-deploy |
    | Re-run `dg ingest deploy --daytona` after restart | ✅ ~5s, same URL + token |
    | Default snapshot resources | 1 CPU / 1 GB / **3 GB disk** |
  - Found + fixed on the way: non-editable installs of this package failed
    (duplicate `force-include` of `dashboard_static` in `pyproject.toml`); every
    pod install had used `pip install -e .`, which masked it.
- [x] **Issues found during the live run (fixed).**
  - Missing `DAYTONA_API_KEY` was detected only after model prep → now checked right after planning.
  - `dg` didn't auto-load `.env` → every `dg` command loads `./.env` then the repo `.env`.
  - Runs list "updated Ns ago" reset every poll (heartbeat time) → "running for 6m" / "finished 3m ago".
  - Run page kept "Initializing Megatron…" through training → "Training — step 2/4 (N rollouts)" from telemetry (`num_steps` in `run.meta`).
  - Redeploy kept old code (pip saw `@main` as satisfied; server never restarted) → pinned to commit, force-reinstall, restart.
  - Restart failed because the old Daytona session had exited (ingest was down ~1 min) → fresh session per start.
  - New pod, run `run_ecc7eb0c…`: ingest vars not in the launch shell → launch silently fell back to a Cloudflare tunnel and never shipped → every launch now prints `ingest: shipping …` or `ingest: OFF …`, half a config fails fast, and `dg ingest push runs/<id>.jsonl` backfills missed runs (idempotent).
  - Unit tests read the developer's `.env` and shipped two fake runs (`run_worker_test`, `run_testgym`) to the real ingest host → tests are hermetic (`DAYTONA_GYM_NO_DOTENV=1` fixture).
- [ ] **Ingest sandbox follow-ups.**
  - [ ] ⚠️ **Priority raised:** on 2026-09-25 the ingest sandbox was found
    **STOPPED with auto-stop = 0** (cause unknown — platform/org limits?). Runs
    kept their data on disk and a redeploy fixed it; launches now warn loudly
    when the ingest host is unreachable. Needs a watchdog / auto-restart.
  - [ ] Auto-restart `dg ingest` after a sandbox restart — try an image
    `ENTRYPOINT` (verify it coexists with Daytona's toolbox daemon); fallback is
    a watchdog that re-runs deploy when `/healthz` fails.
  - [ ] Bigger disk: create from a declarative `Image` with `Resources(disk=…)`
    (snapshot-based create can't set resources; 3 GB ≈ hundreds of runs).
  - [ ] Periodic backup tarball of the data dir to a Daytona Volume (volumes are
    object-storage backed — backup target only, not the live store).
  - [x] Pin to a commit instead of `@main` (`--ref`, resolved via `git ls-remote`).
  - [x] `dg ingest rm <run>…` (token-guarded `DELETE /v1/runs/<id>`); used to remove the two leaked test runs.
- [x] **Phase 2 exit — verified live on RunPod.** Done when (laptop detach item optional):
  - [x] `dg ingest deploy --daytona` (HTTPS preview URL + token), env exported on laptop and pod
  - [x] `python main.py` on the A100 prints the ingest URL (not trycloudflare / 127.0.0.1)
  - [x] dashboard updates live during training from the ingest host; terminal
    status ships (`run_b959a18c…` → completed, 32 rollouts, 2026-09-25)
  - [x] failure path ships too (`run_649d9aca…` → failed: missing `DAYTONA_API_KEY`)
  - [x] terminate the pod mid-run → run stays browsable, shows `worker_lost` within ~90s (A100 pod stopped from the RunPod console, 2026-09-25)
  - [ ] laptop `launch(worker=runpod_worker(), detach=True)` + `run.wait()` works with no PTY sync

---

## Phase 3 — Real BYO portability

Goal: "move off Modal's GPUs to anything" is demonstrated, not claimed.

- [ ] ⏳ **3.1 Worker image contract.** Done when a published image (Slime +
  Megatron + SGLang + `daytona_gym`) runs the gym with documented paths and no
  `/root/slime` assumptions or git clone on the worker
  (`LocalSlimeCompute` defaults, `remote_job.py`).
  - Built: `docker/worker/Dockerfile` = `slimerl/slime@sha256:0589e3b6…` (pinned)
    + the gym wheel pip-installed into Slime's own Python (the adapter runs
    inside Slime's process, so it must share the env). CI:
    `.github/workflows/worker-image.yml` → `ghcr.io/danielgraviet/daytona-gym-worker:{sha,latest}`
    (Docker storage moved to the runner's `/mnt`: the base is 21 GB compressed).
  - Paths come from the worker's env: `DAYTONA_GYM_MODELS_DIR` (checkpoints,
    mount a volume so weights persist), `DAYTONA_GYM_WORKDIR` (`runs/`),
    `SLIME_ROOT` / `MEGATRON_ROOT`; defaults keep the old `/root/…` layout.
    `.dockerignore` keeps `.env` / runs / `.git` out of the image (tested).
  - Base image facts: amd64, Python 3.12, **requires host driver with CUDA ≥ 12.9**.
  - [x] CI build green (~10–30 min; Docker storage on the runner's `/mnt`).
  - [x] Published + public: `ghcr.io/danielgraviet/daytona-gym-worker` and
    `docker.io/dtgraviet/daytona-gym-worker` (`:latest` + `:<sha>`). Docker Hub
    push takes ~13 s (base layers mount from `slimerl/slime`); GHCR re-push ~9 s.
  - [x] Pulled on the home 3090: after the one-time base pull, image updates
    download only the gym layer (~5 s).
  - [ ] ⏳ A full successful run on the 3090 — see 3.4.
- [ ] **3.2 Outbound worker agent** (`daytona_gym/workers/agent.py`). Done when
  `docker run <image> daytona-gym worker --token …` dials out, receives a
  launch payload, reports lifecycle + GPU inventory, and supports safe stop —
  no inbound SSH, no PTY proxy hacks (`SshWorker` stays as an escape hatch).
  - Context: the ingest host from Phase 2 is the natural control plane
    endpoint for the agent to poll.
- [x] **3.3 Dataset/config shipped with the launch payload.** Done when a local
  JSONL not in git trains on the worker (customer note: `git pull` can't see it).
  - Note: local `PromptJsonlDataset` files and local `HarborDataset` task
    folders are embedded in the job (`inline_jsonl`, ≤16 MB, SHA-256 checked)
    and written to `runs/data/inline_<sha>_<name>.jsonl` on the worker; HF and
    Harbor-registry datasets are still fetched on the worker; worker-only
    paths are sent unchanged. Oversized datasets fail with guidance. Job files
    are 0600. PTY uploads are gzip'd and decoded in one python step (found:
    macOS `base64` rejects a file arg, and `base64 | python` hid the failure
    as an empty job file). `tests/contracts/test_dataset_shipping.py`.
  - ⏳ Live check: laptop → pod launch with a JSONL that is not in git.
- [ ] ⏳ **3.4 Dogfood on a second provider** — **the user's home RTX 3090**.
  **Pick up here Monday.**
  - Box: Ubuntu, RTX 3090 24 GB, driver 595 (CUDA 13.2), **16 GB host RAM**
    (15 GiB usable) + 4 GB swap, 16 cores, Wi-Fi (~12 MiB/s), Docker 29.8 +
    NVIDIA Container Toolkit, reachable as `ssh gpu` (Tailscale). Runner script:
    `~/run-gym-3090.sh` (pipes `~/box_worker.py` into the image); secrets in
    `~/.daytona-gym.env` (0600); models in the `daytona-gym-models` volume.
  - What works on the box: image pull, model download + Megatron conversion,
    SGLang, Daytona sandboxes, ingest shipping, Slime perf ingest, 32/32 rollouts.
  - Attempts (2026-09-25), all visible on the ingest dashboard:
    | Run | Settings | Result |
    | --- | --- | --- |
    | `run_89fb7937…` | default colocate (offload to CPU) | **host OOM**: kernel killed the Megatron actor (Ray object store 5.2 GB + CPU offload > 15 GiB) |
    | `run_dca8dbcd…` | no offload, SGLang 30%, 6 turns × 768 | **GPU OOM** in step 0 training: ~5k-token sample's fp32 logits (152k vocab) = 3 GiB |
    | `run_86ac7c56…` | no offload, SGLang 20%, 3 turns × 512 | steps 0–2 trained; **GPU OOM** in step 3 on a 2.8k-token straggler (Megatron grew to 17 GiB) |
    | `run_daf327a3…` | no offload, SGLang 15%, 2 turns × 512 | **all 4 steps trained** (Slime step time 106 → 9.6 → 8.2 → 7.7 s); then a Ray actor died **after training**; kernel log shows host OOM kills in that window |
  - Fixes already shipped from this: Ray object store capped at 2 GiB below
    32 GiB RAM; `offload_train` / `offload_rollout` recipe knobs
    (`--no-offload-*`); launch-time low-RAM warning (quiet when not offloading);
    `box_worker.py` exits with the run's return code.
  - **Monday next steps:**
    1. Confirm what died after step 3: match `journalctl -k` OOM timestamps to
       `~/gym-3090.log` (22:56). Suspect the end-of-run checkpoint save (Slime
       saves at the last rollout; `--save` gathers weights into CPU RAM). Try
       disabling the final save for smoke runs (recipe knob) or saving less.
    2. Product fix for sample length: recipe `tool_output_limit` (tool stdout is
       16,384 chars per observation today) + a per-rollout `max_total_tokens`
       budget that ends the rollout as `truncated` instead of OOMing training.
    3. Turn the working settings into a named preset (e.g. `Box24GB` /
       low-host-RAM recipe) so a 3090/4090 user gets them without tuning.
    4. Optional: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the
       trainer to cut allocator fragmentation (verify SGLang tolerates it).
  - Insight worth keeping: with both engines resident on the GPU, Slime steps
    were **~8–10 s** vs ~44 s on the A100 with offload — the "switching" cost
    disappears for small models.
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

## Ideas surfaced (not scheduled)

- **Sandbox cost + resource usage per rollout from Daytona's own analytics API**
  (`daytona_analytics_api_client`: per-sandbox telemetry metrics/logs/traces and
  usage). Every span already carries `sandbox_id`, so the rollout view could add
  sandbox CPU/mem and $ — "GPU $ idle + sandbox $ per rollout".
- **Daytona GPU sandboxes** exist in the SDK (`Resources(gpu=…, gpu_type=…)`,
  `spot`) — relevant to `PRODUCT_DECISIONS.md` §5 (Daytona GPUs as an optional
  backend under the same `compute` abstraction), after Phase 3.

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
| 2026-09-25 | `7fe0854` | Phase 2: `dg ingest`, worker shipper, `worker_lost`, auth + hardening |
| 2026-09-25 | `2e480a1`+ | `dg ingest deploy --daytona`; wheel build fix; live sandbox probe |
| 2026-09-25 | (1.2) | Slime `perf/*` ingest → trainer vs overhead split; `dg ingest rm` |
| 2026-09-25 | `6492e3b`…`58302b4` | 3.1 worker image (GHCR + Docker Hub), 3.3 dataset shipping, small-RAM fixes; 3090 attempts logged under 3.4 |
