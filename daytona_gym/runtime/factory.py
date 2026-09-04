from __future__ import annotations

from typing import Any

from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.limits import LimitingEnvironmentRuntime, RetryingEnvironmentRuntime


def build_environment_runtime(args: Any) -> EnvironmentRuntime:
    """Resolve the environment runtime from Slime/Daytona args.

    An explicit `args.daytona_environment_runtime` always wins. Otherwise a
    fake runtime is used only when requested; production defaults to the SDK.
    The resolved instance is stored back on `args` so concurrent samples share
    one AsyncDaytona client, retry policy, and concurrency limiter.
    """
    runtime = getattr(args, "daytona_environment_runtime", None)
    wrap = True
    if runtime is not None:
        wrap = bool(getattr(args, "daytona_wrap_runtime", False))
    elif bool(getattr(args, "daytona_use_fake_runtime", False)):
        runtime = FakeEnvironmentRuntime()
    else:
        runtime = DaytonaEnvironmentRuntime.from_args(args)

    if wrap:
        runtime = _apply_policies(runtime, args)
    try:
        args.daytona_environment_runtime = runtime
    except Exception:
        pass
    return runtime


def _apply_policies(runtime: EnvironmentRuntime, args: Any) -> EnvironmentRuntime:
    attempts = int(getattr(args, "daytona_provision_retries", 3) or 1)
    backoff = float(getattr(args, "daytona_provision_retry_backoff_seconds", 0.05) or 0.0)
    if attempts > 1:
        runtime = RetryingEnvironmentRuntime(
            runtime,
            max_attempts=attempts,
            backoff_seconds=backoff,
        )
    max_in_flight = getattr(args, "daytona_max_concurrency", None)
    if max_in_flight is not None:
        runtime = LimitingEnvironmentRuntime(runtime, int(max_in_flight))
    return runtime
