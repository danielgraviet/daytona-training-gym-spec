from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from daytona_gym.runtime.actions import (
    looks_like_hallucinated_tool_result,
    parse_agent_actions,
    sanitize_generation_text,
)
from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.generation import GenerationBackend, GenerationResult
from daytona_gym.runtime.security import DEFAULT_CAPTURE_LIMIT, clip_text
from daytona_gym.runtime.types import (
    EnvironmentHandle,
    EnvironmentSpec,
    ToolAction,
    ToolName,
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
    # Relative paths → file contents written after sandbox create (before generate).
    seed_files: Mapping[str, str] = field(default_factory=dict)
    # If set, run this test command once after seed (before the model speaks).
    # Ensures tool.run_tests appears in traces even when the model skips JSON tools.
    bootstrap_run_tests: str | None = None
    # If True, ignore {"type":"final"} until the latest run_tests exited 0.
    require_passing_tests_for_final: bool = False


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
                    await self._seed_files(env, request, ids)
                    conversation = request.prompt
                    bootstrap_event = await self._bootstrap_run_tests(env, request, ids)
                    if bootstrap_event is not None:
                        events.append(bootstrap_event)
                        conversation = (
                            f"{conversation}\n\n"
                            f"[environment bootstrap run_tests]\n{bootstrap_event.text}\n"
                        )
                    for _turn in range(request.max_turns):
                        generation, gen_event = await self._generate(
                            conversation, request.sampling_params, ids
                        )
                        events.append(gen_event)
                        preview, _ = clip_text(generation.text, 400)
                        model_text = sanitize_generation_text(generation.text)
                        try:
                            actions = parse_agent_actions(generation.text)
                        except DaytonaError as exc:
                            # Bad JSON / unknown shape is an observation, not a job killer.
                            if exc.code is not ErrorCode.USER_CODE_ERROR:
                                raise
                            nudge = (
                                f"\n[harness] {exc.message}. "
                                "Reply with a JSON object, e.g. "
                                '{"type":"tool","name":"read_file","arguments":{"path":"util.py"}} '
                                'or {"type":"tool","name":"write_file",'
                                '"arguments":{"path":"util.py","content":"..."}} '
                                'or {"type":"final","content":"fixed"}.\n'
                            )
                            print(
                                "[daytona-gym] parse user_code_error; nudging model "
                                f"preview={preview!r}",
                                flush=True,
                            )
                            conversation = f"{conversation}{model_text}{nudge}"
                            continue
                        kinds = ",".join(a.parse_kind for a in actions)
                        print(
                            "[daytona-gym] model_turn "
                            f"n={len(actions)} parse=[{kinds}] "
                            f"preview={preview!r}",
                            flush=True,
                        )

                        if looks_like_hallucinated_tool_result(generation.text):
                            nudge = (
                                "\n[harness] Do not invent <tool_result> text. "
                                "Emit a JSON tool call such as "
                                '{"type":"run_tests","arguments":{"command":"python test_broken.py"}} '
                                "or write_file, then wait for the real tool_result.\n"
                            )
                            print(
                                "[daytona-gym] rejected hallucinated tool_result; nudging model",
                                flush=True,
                            )
                            conversation = f"{conversation}{model_text}{nudge}"
                            continue

                        tool_actions = [a for a in actions if not a.is_final]
                        final_actions = [a for a in actions if a.is_final]
                        conversation = f"{conversation}{model_text}"

                        for action in tool_actions:
                            assert action.tool is not None
                            tool_event = await self._execute_tool(
                                env, action.tool, request, ids
                            )
                            events.append(tool_event)
                            conversation = f"{conversation}{tool_event.text}"
                            if (
                                action.tool.name == ToolName.RUN_TESTS
                                and tool_event.ok is False
                            ):
                                conversation += (
                                    "\n[harness] Tests failed. Read the failing files, "
                                    "fix the bug with write_file, then run_tests again. "
                                    "Do not emit final yet.\n"
                                )

                        if final_actions:
                            action = final_actions[-1]
                            if request.require_passing_tests_for_final and not _tests_currently_passing(
                                events
                            ):
                                nudge = (
                                    "\n[harness] Cannot finalize yet: last run_tests did not pass "
                                    "(exit 0 / OK). Fix the code, then call run_tests again "
                                    "(do not invent OK).\n"
                                )
                                print(
                                    "[daytona-gym] rejected premature final; nudging model",
                                    flush=True,
                                )
                                conversation = f"{conversation}{nudge}"
                                continue
                            final_response = action.content
                            status = "completed"
                            break

                        if not tool_actions:
                            # Prose-only / unknown — already appended; ask for JSON.
                            conversation += (
                                "\n[harness] Reply with exactly one JSON tool or final object.\n"
                            )
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
                reward = _reward_from_events(events)
                response_tokens = sum(
                    len(event.token_ids or [])
                    for event in events
                    if event.type == "generation"
                )
                if reward is not None:
                    rollout_span.set_attribute("reward", float(reward))
                rollout_span.set_attribute("response_tokens", int(response_tokens))

        finished_at = _utcnow()
        self._metrics.increment("rollout.count", status=status)
        duration = (finished_at - started_at).total_seconds()
        self._metrics.observe("rollout.duration_seconds", duration, status=status)
        reward = _reward_from_events(events)

        return DaytonaTrajectory(
            run_id=request.run_id,
            rollout_id=request.rollout_id,
            prompt=request.prompt,
            events=events,
            final_response=final_response,
            reward=reward,
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

    async def _seed_files(
        self,
        env: EnvironmentHandle,
        request: RolloutRequest,
        ids: dict[str, object],
    ) -> None:
        if not request.seed_files:
            return
        with self._tracer.span("sandbox.seed", **ids) as span:
            span.set_attribute("file_count", len(request.seed_files))
            for path, content in request.seed_files.items():
                result = await self._runtime.execute(
                    env,
                    ToolAction(
                        name=ToolName.WRITE_FILE,
                        arguments={"path": path, "content": content},
                        timeout_seconds=request.tool_timeout_seconds,
                    ),
                )
                if not result.ok:
                    raise DaytonaError(
                        ErrorCode.TOOL_FAILED,
                        f"seed write failed for {path!r}",
                        details={"path": path, "exit_code": result.exit_code},
                    )

    async def _bootstrap_run_tests(
        self,
        env: EnvironmentHandle,
        request: RolloutRequest,
        ids: dict[str, object],
    ) -> TrajectoryEvent | None:
        command = request.bootstrap_run_tests
        if not command:
            return None
        return await self._execute_tool(
            env,
            ToolAction(
                name=ToolName.RUN_TESTS,
                arguments={"command": command},
                timeout_seconds=request.tool_timeout_seconds,
            ),
            request,
            ids,
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
                preview, truncated = clip_text(generation.text, 512)
                span.set_attribute("generation_preview", preview)
                span.set_attribute("generation_chars", len(generation.text))
                if truncated:
                    span.set_attribute("generation_preview_truncated", True)
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
            token_ids=list(generation.token_ids) if generation.token_ids is not None else None,
            log_probs=list(generation.log_probs) if generation.log_probs is not None else None,
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
                try:
                    result = await self._runtime.execute(env, bounded)
                except DaytonaError as exc:
                    # Invalid model args should become an observation, not kill the job.
                    if exc.code is not ErrorCode.USER_CODE_ERROR:
                        raise
                    status = "error"
                    span.set_attribute("ok", False)
                    span.set_attribute("error_code", str(exc.code))
                    finished = _utcnow()
                    observation = (
                        f'\n<tool_result name="{action.name}" ok="false" '
                        f'exit_code="1" truncated="false">\n'
                        f"[harness] {exc.message}\n"
                        f"Retry with a complete JSON tool call "
                        f'(e.g. write_file needs path+content).\n'
                        f"</tool_result>\n"
                    )
                    return TrajectoryEvent(
                        type="tool",
                        text=observation,
                        started_at=started,
                        finished_at=finished,
                        tool_name=str(action.name),
                        exit_code=1,
                        ok=False,
                        error_code=str(exc.code),
                    )
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


def _tests_currently_passing(events: list[TrajectoryEvent]) -> bool:
    last_tests: TrajectoryEvent | None = None
    for event in events:
        if event.type == "tool" and event.tool_name == "run_tests":
            last_tests = event
    if last_tests is None:
        return False
    return bool(last_tests.ok) and (last_tests.exit_code or 0) == 0


def _reward_from_events(events: list[TrajectoryEvent]) -> float | None:
    """Prefer last run_tests outcome; None if tests never ran."""
    last_tests: TrajectoryEvent | None = None
    for event in events:
        if event.type == "tool" and event.tool_name == "run_tests":
            last_tests = event
    if last_tests is None:
        return None
    if last_tests.ok and (last_tests.exit_code or 0) == 0:
        return 1.0
    return 0.0
