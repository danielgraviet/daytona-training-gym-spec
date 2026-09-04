from __future__ import annotations

from typing import Any

from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.environment import EnvironmentRuntime
from daytona_gym.runtime.fake import FakeEnvironmentRuntime


def build_environment_runtime(args: Any) -> EnvironmentRuntime:
    """Resolve the environment runtime from Slime/Daytona args.

    An explicit `args.daytona_environment_runtime` always wins. Otherwise a
    fake runtime is used only when requested; production defaults to the SDK.
    The resolved instance is stored back on `args` so concurrent samples share
    one AsyncDaytona client.
    """
    runtime = getattr(args, "daytona_environment_runtime", None)
    if runtime is not None:
        return runtime
    if bool(getattr(args, "daytona_use_fake_runtime", False)):
        runtime = FakeEnvironmentRuntime()
    else:
        runtime = DaytonaEnvironmentRuntime.from_args(args)
    try:
        args.daytona_environment_runtime = runtime
    except Exception:
        pass
    return runtime
