# Slime / SGLang / Megatron — learning notes

Goal: talk fluently about this stack in partner and eng conversations — what each piece does, how they fit, what breaks, and what people mean when they say certain words.

Grounded in Daytona Training Gym dogfood on Slime (2026-09), plus how Slime is designed.

---

## 1. The one-sentence picture

**Slime** is an RL post-training *orchestrator*. It does **not** replace the inference engine or the trainer:

| Piece | Job | Analogy |
|---|---|---|
| **SGLang** | Fast inference / sampling (rollouts) | “The model talking” |
| **Megatron-LM** | GPU training (optimizer step, grads) | “The model learning” |
| **Ray** | Process / GPU placement | “The cluster scheduler” |
| **Slime** | Glue: data → rollout → reward → train step → weight sync | “The RL loop conductor” |
| **Daytona (us)** | Sandboxes + tool loop + traces inside *generate* | “The agent’s computer” |

In one training step Slime roughly does:

```text
load prompts
  → SGLang generates (possibly multi-turn / tools)
  → reward model / custom RM scores samples
  → Megatron computes loss (e.g. GRPO) and updates weights
  → push new weights back to SGLang
```

Daytona plugs in at **generate** (`--custom-generate-function-path`), not by forking Megatron.

---

## 2. SGLang — what to know

### What it is

- High-performance LLM **serving / inference** stack (often with a **router** + one or more **engines**).
- Optimized for throughput: continuous batching, prefix cache, structured generation, etc.
- In Slime: used for **rollouts** (sampling trajectories), not for the Adam update.

### Words people use

| Term | Meaning |
|---|---|
| **Router** | HTTP front door; Slime posts to `http://{ip}:{port}/generate` |
| **Engine / server** | Actual GPU process holding the model weights for inference |
| **TP / `rollout-num-gpus-per-engine`** | Tensor parallel width of one inference engine (like `tp_size`) |
| **`mem-fraction-static`** | How much GPU memory SGLang reserves up front (leave room for Megatron if colocated) |
| **`return_logprob` / `output_token_logprobs`** | Per-token log probs needed for RL (importance sampling / TIS) |
| **Colocate** | Train + infer share the same GPUs (Slime `--colocate`); memory juggling matters |
| **Weight sync / update_weights** | After a Megatron step, copy new params into SGLang so the next rollout uses them |

### Common API shape (what custom generate hits)

```http
POST /generate
{
  "text": "<prompt + so-far>",
  "sampling_params": { "temperature": ..., "max_new_tokens": ... },
  "return_logprob": true
}
```

Response (simplified):

```json
{
  "text": "<new tokens as string>",
  "meta_info": {
    "finish_reason": { "type": "stop" | "length" | "abort" },
    "output_token_logprobs": [[logprob, token_id], ...]
  }
}
```

**Critical RL rule:** if you collect logprobs, do **not** post-process the string and re-tokenize — tokens and logprobs must stay aligned. Prefer engine token ids from `output_token_logprobs`.

### Things that break (SGLang side)

- Router IP/port not set yet → generate called before engines are up  
- OOM when colocated with Megatron → lower `--sglang-mem-fraction-static`  
- Missing `output_token_logprobs` when `return_logprob=True` → training can’t trust rollout logprobs  
- Abort / length finish reasons → truncated or aborted samples (status matters for masks)

### Talking points

- “We don’t train inside SGLang; we *sample* there.”  
- “Logprobs from the engine are first-class for off-policy / GRPO-style methods.”  
- “Agent loops (tools, sandboxes) sit in custom generate and still call the same `/generate` endpoint.”

---

## 3. Megatron-LM — what to know

### What it is

- NVIDIA’s framework for **large-model training** (tensor / pipeline / data / context / expert parallel).
- Does **not** natively load Hugging Face `safetensors` the way Transformers does.
- Slime uses Megatron as the **actor (policy) trainer**.

### Words people use

| Term | Meaning |
|---|---|
| **HF checkpoint** | Hugging Face weights + tokenizer (`--hf-checkpoint`) — for tokenizer / convert source |
| **`torch_dist` / distcp** | Megatron’s sharded checkpoint format (preferred; reshard-friendly) |
| **`ref-load`** | Reference policy weights (KL / GRPO baseline), usually same convert as start |
| **TP / PP / CP / EP** | Tensor / pipeline / context / expert parallelism |
| **`--colocate`** | Actor training and rollout inference share GPUs |
| **GRPO / PPO / advantage** | RL algorithm flavor; Slime flags like `--advantage-estimator grpo` |
| **`loss_mask`** | Per-response-token 0/1 — **1 = train on this token**, 0 = ignore (tool obs, padding) |
| **`global-batch-size` / microbatch** | How many samples (or tokens) per optimizer step |

### Checkpoint dance (everyone hits this)

```text
HF download
  → tools/convert_hf_to_torch_dist.py  (+ MODEL_ARGS from scripts/models/*.sh)
  → train with --hf-checkpoint (tokenizer) + --ref-load / --load (Megatron weights)
```

Megatron needs **model shape args** (`--num-layers`, `--hidden-size`, …) to match the architecture. Slime ships these under `scripts/models/`.

### Things that break (Megatron / train side)

- Wrong Megatron path in `PYTHONPATH` (image had `/root/Megatron-LM`, script said `/root/src/Megatron-LM`)  
- HF→dist convert skipped or architecture mismatch  
- **`rotary_base` ≠ HF `rope_theta`** — Slime validates HF config against Megatron args and aborts early:

  ```text
  AssertionError: hf_validate_args failed: rope_theta in hf config 1000000.0
    is not equal to rotary_base 10000, please check the config.
  ```

  Qwen2.5 Instruct checkpoints often use `rope_theta=1000000`. Some Slime `scripts/models/qwen2.5-*.sh` files still ship `--rotary-base 10000`. Fix the model script (or pass `--rotary-base 1000000`) so it matches `config.json`. Dogfood hit this on Qwen2.5-1.5B.  
- Sample fields wrong (`tokens` as strings, `rollout_log_probs` None) → crashes deep in loss (`KL`, advantages) with **opaque TypeErrors**  
- Transformers 5.x: `list(tokenizer(...))` can yield **string keys**, not ids — always prefer `tokenizer.encode(...)`  
- Empty / failed rollouts still fed into train → garbage tensors

### Talking points

- “Megatron owns the optimizer step; HF is usually just the source format + tokenizer.”  
- “RL cares as much about masks and logprobs as about loss formulas.”  
- “If train dies in `compute_advantages_and_returns`, suspect rollout Sample contract first.”

---

## 4. Slime — the conductor

### What it is

- Open RL post-training framework (THUDM) built around **async rollouts** + Megatron + SGLang.
- Exposes **plugin paths** so you don’t fork core for agentic workflows.

### Important hooks (conversation ammo)

| Flag | Use when |
|---|---|
| `--custom-generate-function-path` | Per-sample agent loop (tools, sandbox, multi-turn) — **Daytona’s hook** |
| `--custom-rm-path` | Custom reward (tests passed, verifier, etc.) |
| `--rollout-function-path` | Replace **entire** rollout orchestration (heavier; avoid for MVP) |
| `--prompt-data` + `--input-key` / `--label-key` | JSONL/parquet dataset wiring |
| `--apply-chat-template` | Run HF chat template on prompts |
| `--num-rollout` / `--rollout-batch-size` / `--n-samples-per-prompt` | How much data per cycle |
| `--colocate` | Single-node / shared-GPU dogfood |

Signature people expect for custom generate:

```python
async def generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]:
    ...
```

### Sample contract (memorize this)

A trainable sample roughly needs:

- `tokens: list[int]` — prompt + response ids  
- `response` / `response_length`  
- `loss_mask: list[int]` — length == `response_length`  
- `rollout_log_probs` — aligned with response tokens when used  
- `reward: float`  
- `status` — completed / truncated / aborted / failed  

Tool / environment tokens should be **`loss_mask=0`** (and usually dummy logprobs). Model generations are **`loss_mask=1`**.

### Things that break (Slime / DX)

- **Long fail loop:** engines start before custom generate → bad `DAYTONA_API_KEY` only fails after minutes → then Megatron TypeError. Fix: **preflight** credentials before `ray job submit`.  
- Ray `runtime_env` can **print secrets** in `ray job list` — mitigated via `DAYTONA_API_KEY_FILE`.
- Custom generate import path must be importable on **Ray workers** (`PYTHONPATH`).  
- Small models ignore tool JSON → rollout “succeeds” with `reward=None` and no tool spans.

---

## 5. How Daytona fits (our product angle)

```text
Slime scheduler
  → custom_generate(args, sample, sampling_params)
       → Daytona: provision sandbox
       → optional seed files
       → loop: SGLang /generate ↔ tools (run_tests, write_file, ...)
       → fill Sample (tokens, masks, logprobs, reward hint)
       → telemetry JSONL
  → custom_rm reads metadata / reward
  → Megatron train step
```

**Value prop in one line:** portable rollout environments + useful RL observability, without owning the GPU trainer.

**What we do not do:** build a new RL trainer; fork Slime core for MVP.

---

## 6. Common “syntax” / ops snippets

### Launch shape (conceptual)

```bash
ray start --head --num-gpus N
ray job submit --runtime-env-json='{"env_vars":{"PYTHONPATH":"..."}}' -- \
  python3 train.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate \
    --hf-checkpoint /path/HF/ \
    --ref-load /path/HF_torch_dist/ \
    --prompt-data data.jsonl --input-key prompt --label-key label \
    --custom-generate-function-path mypkg.generate.generate \
    --custom-rm-path mypkg.reward.reward \
    ... MODEL_ARGS / GRPO / SGLANG flags ...
```

### Convert HF → Megatron

```bash
source scripts/models/<model>.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /path/HF \
  --save /path/HF_torch_dist
```

### Daytona dogfood (this repo)

```bash
python -m daytona_gym.preflight          # fail fast on bad API key
bash examples/coding_dogfood/run_on_slime_pod.sh
python -m daytona_gym.telemetry.inspect runs/dogfood.jsonl
```

### Inspect mental model

Healthy coding rollout spans:

```text
rollout
  sandbox.provision
  sandbox.seed
  inference.generate
  tool.run_tests / tool.write_file   # only if the model emits JSON tools
  sandbox.finalize
```

---

## 7. Debugging playbook (what to ask / check)

| Symptom | Likely layer | First checks |
|---|---|---|
| Job dies in 3s with auth | Daytona | `python -m daytona_gym.preflight`; API key / URL |
| 2–3 min then opaque Megatron `TypeError` / KL None | Rollout → train contract | Did generate fail? Empty `tokens` / missing logprobs? |
| `too many dimensions 'str'` | Tokenizer / Sample.tokens | HF encode path; Transformers 5.x `list(batch)` bug |
| OOM on single GPU colocate | SGLang + Megatron | `sglang-mem-fraction-static`, batch sizes, model size |
| `rope_theta … not equal to rotary_base` | Model script vs HF config | Align `--rotary-base` with HF `rope_theta` (often `1000000` for Qwen2.5) |
| Train OK but no tool spans | Model / prompt | Model too small; non-JSON treated as final; check `[daytona-dogfood]` reward |
| Sandbox leak / hang | Env cleanup | finalize on success **and** cancel/timeout; concurrency limiter |
| Import errors on workers | Ray env | `PYTHONPATH` includes Megatron + your package |
| Weight sync weirdness | Slime colocate | Look for `update_weights` / resume memory logs |

**Rule of thumb:** separate “inference up?”, “sandbox up?”, “Sample valid?”, “train step ran?” — don’t debug Megatron when generate never produced ints.

---

## 8. Use cases people discuss

1. **Math / verifiable RL** — generate answer, rule RM (`math`, `deepscaler`), Megatron GRPO. No sandbox.  
2. **Agentic coding RL** — tools + sandbox + test reward (Daytona’s wedge).  
3. **Search / RAG agents** — Slime `search-r1` style custom generate.  
4. **Colocated single-node dogfood** — cheap validation (our A100 path).  
5. **Disaggregated train/infer** — more GPUs: dedicated rollout engines vs actor GPUs (scale story).

When someone says “we’re doing RL on coding agents,” map it to: **who runs tools**, **where logprobs come from**, **what is masked**, **what is reward**.

---

## 9. Sound bites for conversations

- “Slime is orchestration; SGLang is rollout inference; Megatron is the optimizer.”  
- “Agentic RL is mostly a *generate* customization problem, not a new trainer.”  
- “Masks and logprobs are the Sample contract; break those and training fails far from the bug.”  
- “Fail credential checks before you pay for engine startup.”  
- “Small instruct models often won’t emit tool JSON — infra can be fine while reward stays None.”  
- “Daytona’s job is portable environments + traces; BYO GPU and BYO trainer stay first-class.”

---

## 10. Pointers

- Slime usage / Megatron args: https://thudm.github.io/slime/get_started/usage.html  
- Slime customization hooks: https://thudm.github.io/slime/get_started/customization.html  
- This repo: `SLIME_INTEGRATION.md`, `examples/coding_dogfood/FRICTION.md`, `OBSERVABILITY.md`  
- Native agentic example pattern: Slime `examples/search-r1` (custom generate + SGLang `/generate`)

---

*Living doc — update when dogfood surfaces new failure modes (see `FRICTION.md`).*
