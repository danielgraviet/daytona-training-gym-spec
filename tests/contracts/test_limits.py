from __future__ import annotations

import asyncio
from types import SimpleNamespace

from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.factory import build_environment_runtime
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.limits import LimitingEnvironmentRuntime, RetryingEnvironmentRuntime
from daytona_gym.runtime.types import EnvironmentSpec, ToolAction, ToolName


async def test_retry_succeeds_after_transient_provision_failures() -> None:
    inner = FakeEnvironmentRuntime(fail_create_times=2)
    runtime = RetryingEnvironmentRuntime(inner, max_attempts=3, backoff_seconds=0)
    env = await runtime.create(EnvironmentSpec())
    assert env.sandbox_id
    await runtime.close(env)
    assert inner.leaked_sandbox_ids == ()


async def test_retry_gives_up_after_max_attempts() -> None:
    inner = FakeEnvironmentRuntime(fail_create=True)
    runtime = RetryingEnvironmentRuntime(inner, max_attempts=3, backoff_seconds=0)
    try:
        await runtime.create(EnvironmentSpec())
        raise AssertionError("expected provision failure")
    except DaytonaError as exc:
        assert exc.code is ErrorCode.SANDBOX_PROVISION_FAILED
    assert inner.created_ids == ()


async def test_limiter_caps_in_flight_sandboxes() -> None:
    inner = FakeEnvironmentRuntime(create_delay_seconds=0.02)
    runtime = LimitingEnvironmentRuntime(inner, max_in_flight=8)

    async def one() -> None:
        env = await runtime.create(EnvironmentSpec())
        await runtime.execute(
            env,
            ToolAction(name=ToolName.RUN_COMMAND, arguments={"command": "echo hi"}),
        )
        await runtime.close(env)

    await asyncio.gather(*[one() for _ in range(24)])
    assert runtime.peak_in_flight <= 8
    assert runtime.in_flight == 0
    assert inner.leaked_sandbox_ids == ()
    assert len(inner.created_ids) == 24


def test_factory_defaults_to_retrying_sdk_runtime() -> None:
    args = SimpleNamespace()
    runtime = build_environment_runtime(args)
    assert isinstance(runtime, RetryingEnvironmentRuntime)
    assert isinstance(runtime._inner, DaytonaEnvironmentRuntime)
    assert args.daytona_environment_runtime is runtime


def test_factory_fake_flag_is_wrapped_with_retries() -> None:
    args = SimpleNamespace(daytona_use_fake_runtime=True)
    runtime = build_environment_runtime(args)
    assert isinstance(runtime, RetryingEnvironmentRuntime)
    assert isinstance(runtime._inner, FakeEnvironmentRuntime)


def test_factory_limiter_wraps_when_concurrency_set() -> None:
    args = SimpleNamespace(daytona_use_fake_runtime=True, daytona_max_concurrency=4)
    runtime = build_environment_runtime(args)
    assert isinstance(runtime, LimitingEnvironmentRuntime)
