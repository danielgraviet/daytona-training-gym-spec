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
- **Qwen2.5-3B-Instruct multi-turn tools (2026-09-21 H100, after parser fix `59bd246`):**
  `tools run_tests=2, read_file=1, write_file=1` —
  bootstrap → `read_file` → `write_file` → `run_tests` → `final`.
  Previews showed valid tool JSON with trailing `<|im_end|>` (old `json.loads` would have killed these turns).
- **reward=1.0 coding dogfood (job `raysubmit_Ah8AjLWwnxfDK7Qe`):** one generate emitted
  `write_file(a+b)` + `run_tests` + `final`; multi-tool execution (`cd2356b`) ran both tools →
  `status=completed`, `tools run_tests=2, write_file=1`.

## What hurt

1. RunPod `slimerl/slime:latest` exits without keep-alive — need Docker start command `sleep infinity`
2. Smoke script hardcodes `/root/src/Megatron-LM`; image has `/root/Megatron-LM`
3. Must download HF ckpt, convert to `torch_dist`, and fetch data before first run
4. `transformers` 5.x tokenizer callable → string token ids → slime tensorize crash. Fix: `tokenizer.encode(...)`
5. Async JSONL exporter needs explicit `flush()` before Ray workers exit
6. OpenSSH `BatchMode` fails on RunPod (“doesn't support PTY”); interactive SSH works
7. API key in shell history / chat / formerly in `ray job list` `runtime_env` — **mitigated:** launch script writes `/tmp/daytona_gym_api_key` and passes only `DAYTONA_API_KEY_FILE` in Ray runtime_env; still rotate keys that already leaked
8. **Qwen2.5-0.5B / 1.5B appeared to “ignore” JSON tool protocol** → often free text as `final`. Bootstrap still guarantees `tool.run_tests`. **Also harness bugs (2026-09-21):** (a) `parse_agent_action` used `json.loads` on the *entire* string — markdown fences or trailing junk after a valid tool JSON coerced the turn to `final` (same class of issue Slime Search-R1 fixed with stop tags); (b) telemetry stored only `text_chars`, not a generation preview, so we could not see what the model actually emitted. Fixed: fence strip + `raw_decode` first object; log `generation_preview` on spans + `[daytona-gym] model_turn parse=...` prints.
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
13. **3B called tools but wrote the wrong fix (`a - b`) then `final`/`fixed` → reward 0:**
    - Reward path is correct: `infer_reward_from_trajectory` uses **last** `run_tests` ok/exit_code.
    - Likely context bug: dogfood used `--apply-chat-template` (`add_generation_prompt=True`)
      then concatenated raw generations (including `<|im_end|>`) + `<tool_result>` without
      opening a new ChatML user/assistant turn. After the first tool, the prompt is malformed;
      model tends to echo `read_file` contents instead of applying the prompt’s `a + b` example.
    - Mitigation: drop chat template for coding dogfood (plain prompt + concat like Search-R1),
      strip `<|im_end|>` before append, clearer prompt, reject `final` until tests pass.
14. **Model emits `{"type":"write_file",...}` instead of `{"type":"tool","name":"write_file",...}`**
    → was a hard `user_code_error` and failed the Ray job. Parser now accepts tool-name-as-type
    and name-only tool objects (`tool_type_alias` / `tool_name_only`).
15. **Model emitted correct `a + b` write but omitted `path`** (and multi-JSON + echoed `<tool_result>`
    in one turn) → `path is required` crashed the job. Now: default `path=broken.py` /
    `command=python test_broken.py` when omitted; scan multiple JSON objects for first usable
    action; USER_CODE_ERROR on tool exec becomes an observation nudge instead of aborting.
16. **Multi-JSON one turn only executed first tool** → write_file applied `a + b`, but sibling
    `run_tests`/`final` in the same generate were ignored. Model then **hallucinated**
    `<tool_result ...>OK</tool_result>`, we rejected final (no real passing tests), looped to
    `truncated`. Fix: execute **all** tool actions from one turn, then honor final; reject
    hallucinated tool_result markup.

## Concurrency / cleanup (2 sandboxes)

- Job `raysubmit_Nn9JEmvhrNjY88Vt` — pass (see above)

## Bootstrap verified

- Job `raysubmit_qMmVkw1ZeGapWCCi` — `tools run_tests=1`; timeline seed → `tool.run_tests` → generate → finalize

17. **Edge `tool_stall` (2026-09-21 H100):**
    Bootstrap was `python -c "import time; time.sleep(120)"` with
    `DAYTONA_TOOL_TIMEOUT_SECONDS=5`.
    - Before fix (`raysubmit_KA2V83ueywsGuzuV`): `status=failed`, `tool_failed`,
      `run_tests` ~5.32s `! DaytonaError` (SDK: `DaytonaProcessExecutionTimeoutError`
      / "command execution timeout" was mis-mapped).
    - After fix (`raysubmit_HXPwGU2ApTTQna3b`): `status=aborted`,
      `errors tool_timeout=1`, wall ~9.1s, job still fails loudly via
      `_raise_if_unusable`. Finalize/cleanup OK.
18. **Edge `rollout_budget` (2026-09-21 H100, job `raysubmit_qJzB8AvaxQxYU35K`):**
    Default budget was 6s. Actual wall **6.09s** with full success:
    `status=completed`, `reward=1.0`, `run_tests×2 write_file×1`.
    3B coding loop on H100 is fast enough that 6s never trips
    `asyncio.timeout` mid-rollout (finalize sits outside the timeout).
    Recipe default tightened to **3s** so the next run should show
    `rollout_timeout` / `aborted`.

## Missing product pieces (for external partner)

- Harbor adapter (second after Slime)
- Optional: fail-fast hook inside Slime before engine launch (preflight is outside today)
- Shorter inspect UX — shipped `dg` CLI (`8557802`); pull + `pip install -e .` on pod
- Edge recipes — `run_edge_case.sh` (tool_stall / rollout_budget / concurrency_storm / mem_pressure / hard_prompts / wrong_bootstrap)

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
