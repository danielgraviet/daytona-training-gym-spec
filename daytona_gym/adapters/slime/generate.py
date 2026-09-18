from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from daytona_gym.adapters.slime.sample import (
    apply_trajectory,
    attach_cancelled_metadata,
    prompt_text,
    resolve_tokenizer,
)
from daytona_gym.adapters.slime.sglang_generator import SGLangRouterGenerator
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.factory import build_environment_runtime
from daytona_gym.runtime.generation import GenerationBackend
from daytona_gym.runtime.rollout import DaytonaTrajectory, RolloutRequest, RolloutRunner
from daytona_gym.runtime.types import EnvironmentSpec, Tokenizer
from daytona_gym.telemetry.bind import bind_telemetry
from daytona_gym.telemetry.ids import correlation_attributes, new_rollout_id, new_run_id
from daytona_gym.telemetry.metrics import Metrics
from daytona_gym.telemetry.traces import Tracer


async def generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Slime `--custom-generate-function-path` hook.

    Import path: `daytona_gym.adapters.slime.generate`
    """
    runtime = build_environment_runtime(args)
    generator = _resolve_generator(args)
    _store, tracer, metrics = bind_telemetry(args)
    tokenizer: Tokenizer = resolve_tokenizer(args)

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
        worker_id=getattr(args, "daytona_worker_id", None),
        training_step=getattr(args, "daytona_training_step", None),
        rollout_batch_id=getattr(args, "daytona_rollout_batch_id", None),
        seed_files=dict(getattr(args, "daytona_seed_files", None) or {}),
        bootstrap_run_tests=getattr(args, "daytona_bootstrap_run_tests", None),
    )
    runner = RolloutRunner(runtime, generator, tracer=tracer, metrics=metrics)

    try:
        trajectory = await runner.run(request)
    except asyncio.CancelledError:
        attach_cancelled_metadata(sample, run_id=run_id, rollout_id=rollout_id)
        raise

    apply_trajectory(sample, trajectory, tokenizer, args=args)
    await _maybe_reward(args, sample, trajectory, tracer, metrics)
    _flush_telemetry(args)
    _raise_if_unusable(sample, trajectory)
    return sample


def _raise_if_unusable(sample: Any, trajectory: DaytonaTrajectory) -> None:
    """Fail loudly instead of handing Megatron an empty / broken Sample.

    Slime boots SGLang+Megatron before custom generate runs. When sandbox
    provision fails we used to return a hollow sample; training then died
    minutes later with an opaque TypeError in KL/advantages.
    """
    if trajectory.status not in {"failed", "aborted"}:
        return
    code = trajectory.error_code or "platform_error"
    message = trajectory.error_message or "daytona rollout failed"
    try:
        sample.remove_sample = True
    except Exception:
        pass
    raise DaytonaError(
        _error_code(code),
        f"daytona rollout {trajectory.status}: [{code}] {message}",
        details={
            "run_id": trajectory.run_id,
            "rollout_id": trajectory.rollout_id,
            "sandbox_id": trajectory.sandbox_id,
            "status": trajectory.status,
        },
    )


def _error_code(code: str) -> ErrorCode:
    try:
        return ErrorCode(code)
    except ValueError:
        return ErrorCode.PLATFORM_ERROR


def _flush_telemetry(args: Any) -> None:
    exporter = getattr(args, "daytona_telemetry_exporter", None)
    if exporter is None:
        return
    flush = getattr(exporter, "flush", None)
    if callable(flush):
        try:
            flush(timeout=2.0)
        except Exception:
            return


async def _maybe_reward(
    args: Any,
    sample: Any,
    trajectory: DaytonaTrajectory,
    tracer: Tracer,
    metrics: Metrics,
) -> None:
    reward_fn = getattr(args, "daytona_reward_function", None)
    if reward_fn is None:
        return
    ids = correlation_attributes(
        run_id=trajectory.run_id,
        rollout_id=trajectory.rollout_id,
        project_id=getattr(args, "daytona_project_id", None),
        sample_id=trajectory.sample_id,
        sandbox_id=trajectory.sandbox_id,
        worker_id=getattr(args, "daytona_worker_id", None),
        training_step=getattr(args, "daytona_training_step", None),
        rollout_batch_id=getattr(args, "daytona_rollout_batch_id", None),
    )
    started = time.perf_counter()
    status = "ok"
    try:
        with tracer.span("reward.compute", **ids):
            result = reward_fn(args, sample)
            if inspect.isawaitable(result):
                result = await result
            reward = float(result)
            sample.reward = reward
            trajectory.reward = reward
            sample.metadata.setdefault("daytona", {})["reward"] = reward
    except DaytonaError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise DaytonaError(ErrorCode.REWARD_FAILED, f"reward failed: {exc}") from exc
    finally:
        metrics.observe(
            "reward.duration_seconds",
            time.perf_counter() - started,
            status=status,
        )


def _resolve_generator(args: Any) -> GenerationBackend:
    generator = getattr(args, "daytona_generator", None)
    if generator is not None:
        return generator
    generator = SGLangRouterGenerator.from_args(args)
    try:
        args.daytona_generator = generator
    except Exception:
        pass
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
