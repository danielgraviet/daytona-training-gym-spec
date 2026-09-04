from __future__ import annotations

import inspect

from daytona_gym.adapters.slime import generate


def test_generate_hook_is_async_and_importable() -> None:
    assert inspect.iscoroutinefunction(generate)
    params = list(inspect.signature(generate).parameters)
    assert params == ["args", "sample", "sampling_params"]
