# Dogfood friction log

## Environment

- Date: 2026-09-18
- GPU / machine: RunPod A100-SXM4-80GB (`slimerl/slime:latest`, start cmd `sleep infinity`, high SHM)
- Slime: image-bundled `/root/slime` + `/root/Megatron-LM`
- Model: Qwen2.5-0.5B-Instruct (HF + `torch_dist` convert)
- Data: math smoke (`dapo-math-17k`) then coding (`examples/coding_dogfood/prompts/coding_one.jsonl`)
- Daytona: default python snapshot + in-process seed files (no custom coding image)

## What worked

- Vanilla smoke → Megatron train step
- Daytona custom generate + RM → train step
- Coding dogfood with `sandbox.seed` + inspect timeline
- Concurrency=2: both sandboxes provisioned, seeded, finalized (`raysubmit_Nn9JEmvhrNjY88Vt`)
- Bootstrap `tool.run_tests` after seed (`raysubmit_qMmVkw1ZeGapWCCi`, inspect: seed → `tool.run_tests` → generate → finalize; `tools run_tests=1`)

## What hurt

1. RunPod `slimerl/slime:latest` exits without keep-alive — need Docker start command `sleep infinity`
2. Smoke script hardcodes `/root/src/Megatron-LM`; image has `/root/Megatron-LM`
3. Must download HF ckpt, convert to `torch_dist`, and fetch data before first run
4. `transformers` 5.x tokenizer callable → string token ids → slime tensorize crash. Fix: `tokenizer.encode(...)`
5. Async JSONL exporter needs explicit `flush()` before Ray workers exit
6. OpenSSH `BatchMode` fails on RunPod (“doesn't support PTY”); interactive SSH works
7. API key in shell history / chat / `ray job list` `runtime_env` — rotate after dogfood
8. **Qwen2.5-0.5B / 1.5B ignore JSON tool protocol** → model emits free text as `final`, so no model-driven `tool.*` spans and previously `reward=None`. Mitigation: `daytona_bootstrap_run_tests` (default on in `generate_dogfood`) runs `python test_broken.py` once after seed so traces always include `tool.run_tests` and reward is at least `0.0`.
9. **Bootstrap via `args` setattr alone can silently no-op on Ray workers** — jobs showed `sandbox.seed` but skipped `tool.run_tests` until `DAYTONA_BOOTSTRAP_RUN_TESTS` / `DAYTONA_BOOTSTRAP_RUN_TESTS_CMD` / `DAYTONA_SEED_CODING` were injected in Ray `runtime_env` and resolved inside `generate.py` (`f97d2a9`). Prefer env for anything the worker must see.
10. Pod `git pull` blocked by leftover ad-hoc patches — `git checkout -- . && git clean -fd`
11. **Bad Daytona API key fails late and opaquely (critical DX):**
    - Slime still boots Ray + SGLang + Megatron (~2–3 min) before custom generate runs
    - Only then sandbox provision fails
    - We used to return a hollow Sample; Megatron died with
      `TypeError: 'NoneType' object is not iterable` in `compute_advantages_and_returns` (KL)
    - Dev sees a training crash, not `sandbox_provision_failed` / unauthorized
    - Mitigation shipped: `python -m daytona_gym.preflight` before Slime boot in
      `run_on_slime_pod.sh`, and generate now **raises** a clear `DaytonaError` on
      failed/aborted trajectories instead of feeding Megatron empty tensors
12. Slime `scripts/models/qwen2.5-1.5B.sh` shipped `--rotary-base 10000` but HF
    `rope_theta=1000000` → `hf_validate_args` AssertionError until patched

## Concurrency / cleanup (2 sandboxes)

- Job `raysubmit_Nn9JEmvhrNjY88Vt` — pass (see above)

## Bootstrap verified

- Job `raysubmit_qMmVkw1ZeGapWCCi` — `tools run_tests=1`; timeline seed → `tool.run_tests` → generate → finalize

## Missing product pieces (for external partner)

- Tool-loop on a model that emits JSON tools (≥4B) — bootstrap covers forced first `run_tests` today
- Redact secrets from Ray runtime_env dumps / docs warning
- Optional: fail-fast hook inside Slime before engine launch (preflight is outside today)

## Errors / stack traces worth keeping

```text
ValueError: too many dimensions 'str'
  at slime.observability.rollout_data_utils._cpu_tensor
```

```text
TypeError: 'NoneType' object is not iterable
  at slime.backends.megatron_utils.loss.compute_advantages_and_returns
  (bad DAYTONA_API_KEY → empty rollout sample → KL over None)
  job: raysubmit_2HR6k8zxY5zZh8g8
```

```text
[daytona-dogfood] status=completed sandbox=49f58372-... reward=None tokens=200
```
