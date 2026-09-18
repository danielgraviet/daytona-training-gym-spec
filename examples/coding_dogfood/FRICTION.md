# Dogfood friction log

## Environment

- Date: 2026-09-18
- GPU / machine: RunPod A100-SXM4-80GB (`slimerl/slime:latest`, start cmd `sleep infinity`, high SHM)
- Slime: image-bundled `/root/slime` + `/root/Megatron-LM`
- Model: Qwen2.5-0.5B-Instruct (HF + `torch_dist` convert)
- Data: math smoke (`dapo-math-17k`) then coding (`examples/coding_dogfood/prompts/coding_one.jsonl`)
- Daytona: default python snapshot + in-process seed files (no custom coding image)

## What worked

- Vanilla smoke: `scripts/run-qwen2.5-0.5B-gb10-smoke.sh` (Megatron path fixed to `/root/Megatron-LM`) → job succeeded
- Daytona hooks: `--custom-generate-function-path` + `--custom-rm-path` → one train step succeeded
- Coding dogfood: `examples/coding_dogfood/run_on_slime_pod.sh` → job `raysubmit_EX8vpma9waRiw9WD` succeeded
- Traces show `sandbox.provision` → `sandbox.seed` (~0.6s) → `inference.generate` → `sandbox.finalize`
- Sandbox id logged: `49f58372-2de5-47ff-a76a-3f2fe7f5f324`, status=`completed`, tokens=200

## What hurt

1. RunPod `slimerl/slime:latest` exits without keep-alive — need Docker start command `sleep infinity`
2. Smoke script hardcodes `/root/src/Megatron-LM`; image has `/root/Megatron-LM`
3. Must download HF ckpt, convert to `torch_dist`, and fetch data before first run
4. `transformers` 5.x: calling the HF tokenizer as a callable and `list(batch)` yields string keys → Sample.tokens become strings → slime tensorize crashes (`ValueError: too many dimensions 'str'`). Fix: use `tokenizer.encode(...)`
5. Async JSONL exporter needs an explicit `flush()` before Ray workers exit or the file stays empty
6. OpenSSH `BatchMode` remote commands fail on RunPod (“doesn't support PTY”); interactive SSH works
7. API key ended up in shell history / chat — rotate after dogfood
8. **Qwen2.5-0.5B does not follow the JSON tool protocol** on the coding prompt. Non-JSON text is treated as `final`, so there are no `tool.run_tests` / `tool.write_file` spans. `reward=None` (never ran tests). Seed + train still work; real tool-loop needs a larger instruct model (≥1.5B/4B) or a forced/scripted tool path.
9. Pod `git pull` blocked by leftover ad-hoc patches (`sample.py`, `generate_dogfood.py`) — need `git checkout -- . && git clean -fd` before pull

## Missing product pieces (for external partner)

- One-command dogfood script shipped (`run_on_slime_pod.sh`) — still assumes fixed `/root/...` paths
- Tool-loop validation on a model that actually emits JSON tools (0.5B is insufficient)
- Optional “force first tool turn” / verifier bootstrap so small models still exercise `run_tests`
- Concurrency / leak check with `daytona_max_concurrency=2` (next GPU check)

## Errors / stack traces worth keeping

```text
ValueError: too many dimensions 'str'
  at slime.observability.rollout_data_utils._cpu_tensor
  cause: Sample.tokens held tokenizer BatchEncoding keys (str) under transformers 5.x
```

```text
[daytona-dogfood] status=completed sandbox=49f58372-... reward=None tokens=200
  # coding prompt; no tool turns on 0.5B
```
