from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from daytona_gym.runtime.actions import parse_agent_action
from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.generation import GenerationBackend, GenerationResult
from daytona_gym.runtime.security import DEFAULT_CAPTURE_LIMIT, clip_text
from daytona_gym.runtime.types import (
    EnvironmentHandle,
    EnvironmentSpec,
    ToolAction,
    ToolResult,
    TrajectoryEvent,
)
from daytona_gym.telemetry.ids import correlation_attributes
from daytona_gym.telemetry.metrics import Metrics, NoOpMetrics
from daytona_gym.telemetry.traces import NoOpTracer, Tracer


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    sandbox_id: str | None = None
    sample_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RolloutRequest:
    run_id: str
    rollout_id: str
    prompt: str
    spec: EnvironmentSpec
    sampling_params: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    max_turns: int = 8
    sample_id: str | None = None
    project_id: str | None = None
    stdout_limit: int = DEFAULT_CAPTURE_LIMIT
    tool_timeout_seconds: float | None = None
    worker_id: str | None = None
    training_step: int | str | None = None
    rollout_batch_id: str | None = None


class RolloutRunner:
    """Framework-neutral agent loop over an EnvironmentRuntime."""

    def __init__(
        self,
        runtime: EnvironmentRuntime,
        generator: GenerationBackend,
        *,
        tracer: Tracer | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self._runtime = runtime
        self._generator = generator
        self._tracer = tracer if tracer is not None else NoOpTracer()
        self._metrics = metrics if metrics is not None else NoOpMetrics()

    async def run(self, request: RolloutRequest) -> DaytonaTrajectory:
        started_at = _utcnow()
        events: list[TrajectoryEvent] = []
        env: EnvironmentHandle | None = None
        status = "completed"
        error_code: str | None = None
        error_message: str | None = None
        final_response: str | None = None
        sandbox_id: str | None = None
        ids: dict[str, object] = correlation_attributes(
            run_id=request.run_id,
            rollout_id=request.rollout_id,
            project_id=request.project_id,
            sample_id=request.sample_id,
            worker_id=request.worker_id,
            training_step=request.training_step,
            rollout_batch_id=request.rollout_batch_id,
        )

        with self._tracer.span("rollout", **ids) as rollout_span:
            try:
                async with asyncio.timeout(request.timeout_seconds):
                    env = await self._provision(request, ids)
                    sandbox_id = env.sandbox_id
                    ids = {**ids, "sandbox_id": sandbox_id}
                    rollout_span.set_attribute("sandbox_id", sandbox_id)
                    conversation = request.prompt
                    for _turn in range(request.max_turns):
                        generation, gen_event = await self._generate(
                            conversation, request.sampling_params, ids
                        )
                        events.append(gen_event)
                        action = parse_agent_action(generation.text)
                        if action.is_final:
                            final_response = action.content
                            status = "completed"
                            break
                        assert action.tool is not None
                        tool_event = await self._execute_tool(
                            env, action.tool, request, ids
                        )
                        events.append(tool_event)
                        conversation = f"{conversation}{generation.text}{tool_event.text}"
                    else:
                        status = "truncated"
            except TimeoutError:
                status = "aborted"
                error_code = str(ErrorCode.ROLLOUT_TIMEOUT)
                error_message = "rollout exceeded timeout"
            except DaytonaError as exc:
                status = exc.rollout_status
                error_code = str(exc.code)
                error_message = exc.message
                events.append(
                    TrajectoryEvent(
                        type="error",
                        text=exc.message,
                        started_at=_utcnow(),
                        finished_at=_utcnow(),
                        error_code=str(exc.code),
                    )
                )
            except asyncio.CancelledError:
                status = "aborted"
                error_message = "rollout cancelled"
                self._metrics.increment("rollout.count", status="aborted")
                raise
            except Exception as exc:
                status = "failed"
                error_code = str(ErrorCode.PLATFORM_ERROR)
                error_message = str(exc)
                events.append(
                    TrajectoryEvent(
                        type="error",
                        text=str(exc),
                        started_at=_utcnow(),
                        finished_at=_utcnow(),
                        error_code=str(ErrorCode.PLATFORM_ERROR),
                    )
                )
            finally:
                if env is not None:
                    finalize_ids = {**ids, "sandbox_id": env.sandbox_id}
                    with self._tracer.span("sandbox.finalize", **finalize_ids):
                        try:
                            await self._runtime.close(env)
                        except Exception:
                            pass
                rollout_span.set_attribute("status", status)
                if error_code is not None:
                    rollout_span.set_attribute("error_code", error_code)

        finished_at = _utcnow()
        self._metrics.increment("rollout.count", status=status)
        duration = (finished_at - started_at).total_seconds()
        self._metrics.observe("rollout.duration_seconds", duration, status=status)

        return DaytonaTrajectory(
            run_id=request.run_id,
            rollout_id=request.rollout_id,
            prompt=request.prompt,
            events=events,
            final_response=final_response,
            reward=None,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            sandbox_id=sandbox_id,
            sample_id=request.sample_id,
            error_code=error_code,
            error_message=error_message,
        )

    async def _provision(
        self,
        request: RolloutRequest,
        ids: dict[str, object],
    ) -> EnvironmentHandle:
        started = time.perf_counter()
        status = "ok"
        try:
            with self._tracer.span("sandbox.provision", **ids):
                try:
                    async with asyncio.timeout(request.spec.timeout_seconds):
                        return await self._runtime.create(request.spec)
                except TimeoutError as exc:
                    raise DaytonaError(
                        ErrorCode.SANDBOX_TIMEOUT,
                        "sandbox provision timed out",
                    ) from exc
        except Exception:
            status = "error"
            raise
        finally:
            self._metrics.observe(
                "sandbox.startup_seconds",
                time.perf_counter() - started,
                status=status,
            )

    async def _generate(
        self,
        conversation: str,
        sampling_params: dict[str, Any],
        ids: dict[str, object],
    ) -> tuple[GenerationResult, TrajectoryEvent]:
        started = _utcnow()
        mono = time.perf_counter()
        status = "ok"
        try:
            with self._tracer.span("inference.generate", **ids) as span:
                try:
                    generation = await self._generator.generate(conversation, sampling_params)
                except DaytonaError:
                    raise
                except Exception as exc:
                    raise DaytonaError(
                        ErrorCode.INFERENCE_FAILED,
                        f"inference failed: {exc}",
                    ) from exc
                if generation.request_id:
                    span.set_attribute("model_request_id", generation.request_id)
        except Exception:
            status = "error"
            raise
        finally:
            self._metrics.observe(
                "inference.duration_seconds",
                time.perf_counter() - mono,
                status=status,
            )
        finished = _utcnow()
        event = TrajectoryEvent(
            type="generation",
            text=generation.text,
            started_at=started,
            finished_at=finished,
        )
        return generation, event

    async def _execute_tool(
        self,
        env: EnvironmentHandle,
        action: ToolAction,
        request: RolloutRequest,
        ids: dict[str, object],
    ) -> TrajectoryEvent:
        started = _utcnow()
        timeout = action.timeout_seconds
        if timeout is None:
            timeout = request.tool_timeout_seconds
        bounded = ToolAction(
            name=action.name,
            arguments=action.arguments,
            timeout_seconds=timeout,
        )
        tool = str(action.name)
        mono = time.perf_counter()
        status = "ok"
        try:
            with self._tracer.span(f"tool.{action.name}", **ids) as span:
                span.set_attribute("tool", tool)
                result = await self._runtime.execute(env, bounded)
                span.set_attribute("ok", result.ok)
                if result.exit_code is not None:
                    span.set_attribute("exit_code", result.exit_code)
                if not result.ok:
                    status = "error"
        except Exception:
            status = "error"
            raise
        finally:
            self._metrics.observe(
                "tool.duration_seconds",
                time.perf_counter() - mono,
                tool=tool,
                status=status,
            )
        finished = _utcnow()
        observation = format_observation(bounded, result, stdout_limit=request.stdout_limit)
        return TrajectoryEvent(
            type="tool",
            text=observation,
            started_at=started,
            finished_at=finished,
            tool_name=str(action.name),
            exit_code=result.exit_code,
            ok=result.ok,
        )


def format_observation(
    action: ToolAction,
    result: ToolResult,
    *,
    stdout_limit: int = DEFAULT_CAPTURE_LIMIT,
) -> str:
    stdout, stdout_truncated = clip_text(result.stdout, stdout_limit)
    stderr, stderr_truncated = clip_text(result.stderr, stdout_limit)
    truncated = result.truncated or stdout_truncated or stderr_truncated
    parts = [
        f'\n<tool_result name="{action.name}" ok="{str(result.ok).lower()}" '
        f'exit_code="{result.exit_code if result.exit_code is not None else ""}" '
        f'truncated="{str(truncated).lower()}">',
        stdout.rstrip("\n"),
    ]
    if stderr:
        parts.append(stderr.rstrip("\n"))
    parts.append("</tool_result>\n")
    return "\n".join(parts)
