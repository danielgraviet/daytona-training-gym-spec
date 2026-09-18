# Dogfood friction log

## Environment

- Date: 2026-09-18
- GPU / machine: RunPod A100-SXM4-80GB (`slimerl/slime:latest`, start cmd `sleep infinity`, high SHM)
- Slime: image-bundled `/root/slime` + `/root/Megatron-LM`
- Model: Qwen2.5-0.5B-Instruct (HF + `torch_dist` convert)
- Data: `zhuzilin/dapo-math-17k`
- Daytona: default python snapshot (no custom coding image)

## What worked

- Vanilla smoke: `scripts/run-qwen2.5-0.5B-gb10-smoke.sh` (Megatron path fixed to `/root/Megatron-LM`) → job succeeded
- Daytona hooks: `--custom-generate-function-path` + `--custom-rm-path` → one train step succeeded
- Traces: `runs/dogfood.jsonl` inspect shows `rollout` → `sandbox.provision` → `inference.generate` → `sandbox.finalize`

## What hurt

1. RunPod `slimerl/slime:latest` exits without keep-alive — need Docker start command `sleep infinity`
2. Smoke script hardcodes `/root/src/Megatron-LM`; image has `/root/Megatron-LM`
3. Must download HF ckpt, convert to `torch_dist`, and fetch dapo data before first run
4. `transformers` 5.x: calling the HF tokenizer as a callable and `list(batch)` yields string keys → Sample.tokens become strings → slime tensorize crashes (`ValueError: too many dimensions 'str'`). Fix: use `tokenizer.encode(...)`
5. Async JSONL exporter needs an explicit `flush()` before Ray workers exit or the file stays empty
6. OpenSSH `BatchMode` remote commands fail on RunPod (“doesn't support PTY”); interactive SSH works
7. API key ended up in shell history / chat — rotate after dogfood

## Missing product pieces (for external partner)

- Documented one-command dogfood script (Megatron path + Daytona env + flush)
- Coding snapshot/image with tiny failing pytest task (math prompts don't exercise tools)
- Launch glue still manual (`generate_dogfood.py` wrapper for env → args)

## Errors / stack traces worth keeping

```text
ValueError: too many dimensions 'str'
  at slime.observability.rollout_data_utils._cpu_tensor
  cause: Sample.tokens held tokenizer BatchEncoding keys (str) under transformers 5.x
```
