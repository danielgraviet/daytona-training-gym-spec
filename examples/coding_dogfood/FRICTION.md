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

## What hurt

1. RunPod `slimerl/slime:latest` exits without keep-alive — need Docker start command `sleep infinity`
2. Smoke script hardcodes `/root/src/Megatron-LM`; image has `/root/Megatron-LM`
3. Must download HF ckpt, convert to `torch_dist`, and fetch data before first run
4. `transformers` 5.x tokenizer callable → string token ids → slime tensorize crash. Fix: `tokenizer.encode(...)`
5. Async JSONL exporter needs explicit `flush()` before Ray workers exit
6. OpenSSH `BatchMode` fails on RunPod (“doesn't support PTY”); interactive SSH works
7. API key in shell history / chat / `ray job list` `runtime_env` — rotate after dogfood
8. **Qwen2.5-0.5B ignores JSON tool protocol** → no `tool.*` spans, `reward=None`
9. Pod `git pull` blocked by leftover ad-hoc patches — `git checkout -- . && git clean -fd`
10. **Bad Daytona API key fails late and opaquely (critical DX):**
    - Slime still boots Ray + SGLang + Megatron (~2–3 min) before custom generate runs
    - Only then sandbox provision fails
    - We used to return a hollow Sample; Megatron died with
      `TypeError: 'NoneType' object is not iterable` in `compute_advantages_and_returns` (KL)
    - Dev sees a training crash, not `sandbox_provision_failed` / unauthorized
    - Mitigation shipped: `python -m daytona_gym.preflight` before Slime boot in
      `run_on_slime_pod.sh`, and generate now **raises** a clear `DaytonaError` on
      failed/aborted trajectories instead of feeding Megatron empty tensors

## Concurrency / cleanup (2 sandboxes)

- Job `raysubmit_Nn9JEmvhrNjY88Vt` — pass (see above)

## Missing product pieces (for external partner)

- Tool-loop on a model that emits JSON tools (≥1.5B/4B) or forced first `run_tests`
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
