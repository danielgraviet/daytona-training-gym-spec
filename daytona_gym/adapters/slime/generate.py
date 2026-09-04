from __future__ import annotations

import asyncio
from typing import Any

from daytona_gym.adapters.slime.sample import (
    apply_trajectory,
    attach_cancelled_metadata,
    prompt_text,
)
from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import GenerationBackend
from daytona_gym.runtime.rollout import RolloutRequest, RolloutRunner
from daytona_gym.runtime.types import EnvironmentSpec, OrdinalTokenizer, Tokenizer
from daytona_gym.telemetry.ids import new_rollout_id, new_run_id
from daytona_gym.telemetry.metrics import Metrics, NoOpMetrics
from daytona_gym.telemetry.traces import NoOpTracer, Tracer


async def generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Slime `--custom-generate-function-path` hook.

    Import path: `daytona_gym.adapters.slime.generate`
    """
    runtime = _require_runtime(args)
    generator = _require_generator(args)
    tracer: Tracer = getattr(args, "daytona_tracer", None) or NoOpTracer()
    metrics: Metrics = getattr(args, "daytona_metrics", None) or NoOpMetrics()
    tokenizer: Tokenizer = getattr(args, "daytona_tokenizer", None) or OrdinalTokenizer()

    run_id = str(getattr(args, "daytona_run_id", None) or new_run_id())
    rollout_id = _daytona_rollout_id(sample)
    sample_id = _sample_id(sample)

    spec = EnvironmentSpec(
        image=getattr(args, "daytona_image", None),
        snapshot=getattr(args, "daytona_snapshot", None),
        timeout_seconds=getattr(args, "daytona_sandbox_timeout_seconds", None),
        metadata={
            "run_id": run_id,
            "rollout_id": rollout_id,
            **_string_metadata(getattr(args, "daytona_env_metadata", {}) or {}),
        },
    )
    request = RolloutRequest(
        run_id=run_id,
        rollout_id=rollout_id,
        prompt=prompt_text(sample),
        spec=spec,
        sampling_params=dict(sampling_params or {}),
        timeout_seconds=getattr(args, "daytona_timeout_seconds", None),
        max_turns=int(getattr(args, "daytona_max_turns", 8)),
        sample_id=sample_id,
        project_id=getattr(args, "daytona_project_id", None),
        stdout_limit=int(getattr(args, "daytona_stdout_limit", 16_384)),
        tool_timeout_seconds=getattr(args, "daytona_tool_timeout_seconds", None),
    )
    runner = RolloutRunner(runtime, generator, tracer=tracer, metrics=metrics)

    try:
        trajectory = await runner.run(request)
    except asyncio.CancelledError:
        attach_cancelled_metadata(sample, run_id=run_id, rollout_id=rollout_id)
        raise

    apply_trajectory(sample, trajectory, tokenizer)
    return sample


def _require_runtime(args: Any) -> EnvironmentRuntime:
    runtime = getattr(args, "daytona_environment_runtime", None)
    if runtime is not None:
        return runtime
    if bool(getattr(args, "daytona_use_fake_runtime", False)):
        return FakeEnvironmentRuntime()
    raise DaytonaError(
        ErrorCode.PLATFORM_ERROR,
        "no environment runtime configured; set args.daytona_environment_runtime",
    )


def _require_generator(args: Any) -> GenerationBackend:
    generator = getattr(args, "daytona_generator", None)
    if generator is None:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            "no generation backend configured; set args.daytona_generator",
        )
    return generator


def _daytona_rollout_id(sample: Any) -> str:
    rollout_id = getattr(sample, "rollout_id", None)
    if rollout_id is not None:
        return f"rollout_{rollout_id}"
    index = getattr(sample, "index", None)
    if index is not None:
        return f"rollout_{index}"
    return new_rollout_id()


def _sample_id(sample: Any) -> str | None:
    index = getattr(sample, "index", None)
    if index is None:
        return None
    return str(index)


def _string_metadata(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}
