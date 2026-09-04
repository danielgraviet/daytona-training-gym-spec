# Slime Integration

## Why Slime first

Slime is a good first integration because its rollout path is designed around asynchronous generation and exposes clean customization hooks for agentic workflows.

The integration should preserve Slime's outer scheduling whenever possible.

## Preferred hook

Use:

```text
--custom-generate-function-path
```

for the Daytona agent loop.

Current Slime guidance explicitly positions this hook for:

- tool calls,
- sandbox execution,
- multi-turn generation,
- browser/terminal interaction,
- custom agent loops.

Do **not** replace the whole rollout function initially.

Use:

```text
--rollout-function-path
```

only if Daytona later needs control over task sampling, buffering, requeueing, or the outer rollout scheduler in ways Slime cannot support.

## Reward hook

Prefer keeping reward semantics outside Daytona core.

Use Slime's:

```text
--custom-rm-path
```

for verifier/test-based reward logic when appropriate.

Daytona may provide helpers such as:

```python
def tests_passed_reward(sandbox, test_command: str) -> float:
    ...
```

but should not make reward computation a hard-coded platform concern.

## Adapter shape

Pseudo-code:

```python
async def daytona_generate(args, sample, sampling_params):
    ctx = start_rollout_trace(sample)

    env = await runtime.create_environment(
        image=args.daytona_image,
        snapshot=args.daytona_snapshot,
        metadata={"rollout_id": sample.rollout_id},
    )

    try:
        conversation = sample.prompt

        while True:
            generation = await generate_with_sglang(
                conversation,
                sampling_params,
            )

            record_generation_metrics(ctx, generation)

            action = parse_agent_action(generation)

            if action.is_final:
                break

            result = await runtime.execute_tool(env, action)
            record_tool_span(ctx, action, result)
            conversation = append_observation(conversation, result)

        sample = attach_tokens_and_response(sample, conversation)
        sample.status = COMPLETED
        return sample

    except RolloutTimeout:
        sample.status = ABORTED
        return sample

    except Exception as exc:
        sample.status = FAILED
        attach_error(sample, exc)
        return sample

    finally:
        await runtime.finalize(env)
        finish_rollout_trace(ctx)
```

This is conceptual. Match actual Slime `Sample` fields and tokenization behavior precisely in implementation.

## Sample contract

The adapter must preserve Slime's expected sample semantics.

Important fields include:

- token sequence,
- response length,
- response text,
- reward,
- status,
- optional loss mask,
- optional rollout log probabilities,
- metadata used by downstream training.

Do not create a Daytona-specific sample format inside the Slime adapter. Convert from Daytona's framework-neutral trajectory object into Slime's native `Sample` at the boundary.

## Suggested internal trajectory object

```python
@dataclass
class DaytonaTrajectory:
    run_id: str
    rollout_id: str
    prompt: str
    events: list[TrajectoryEvent]
    final_response: str | None
    reward: float | None
    status: str
    started_at: datetime
    finished_at: datetime | None
```

Slime-specific token IDs and loss masks should be attached in adapter code, not embedded into every Daytona runtime object.

## Preserve async execution

The custom generation function should be `async` and must avoid blocking the event loop on sandbox calls.

All Daytona SDK methods used in rollout execution should expose async versions.

Bad:

```python
result = sandbox.exec_sync(cmd)
```

Preferred:

```python
result = await sandbox.exec(cmd)
```

## Failures

Map errors into useful categories:

```text
sandbox_provision_failed
sandbox_timeout
tool_timeout
tool_nonzero_exit
inference_timeout
rollout_deadline_exceeded
reward_failed
user_code_error
platform_error
```

Keep Slime's native rollout status (`COMPLETED`, `TRUNCATED`, `ABORTED`, `FAILED`) while attaching a more specific Daytona reason in metadata.

## Contract tests

Before running GPUs, create CPU-only tests for:

- custom hook imports correctly,
- adapter accepts a Slime `Sample`,
- sample identifiers are preserved,
- successful rollout returns valid sample,
- sandbox failure maps to valid status,
- timeout maps to valid status,
- metadata is serializable,
- traces always close,
- sandbox cleanup runs on exceptions.

Slime itself uses plugin contract tests for custom hook shapes. Follow the same philosophy.

## References

Official Slime docs to keep nearby:

- Customization Guide: https://thudm.github.io/slime/get_started/customization.html
- Usage Guide: https://thudm.github.io/slime/get_started/usage.html
- Observability: https://thudm.github.io/slime/advanced/observability.html
