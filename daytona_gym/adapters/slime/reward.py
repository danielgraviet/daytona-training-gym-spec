from __future__ import annotations

from typing import Any

from daytona_gym.runtime.reward import tests_passed_reward as _tests_passed_reward

__all__ = ["reward", "tests_passed_reward", "custom_rm"]

tests_passed_reward = _tests_passed_reward


async def reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """Slime `--custom-rm-path` entrypoint.

    Prefer reward already attached during generate (sandbox-side tests). Fall
    back to label heuristics only when no Daytona reward was recorded.
    """
    del kwargs
    meta = getattr(sample, "metadata", None) or {}
    daytona = meta.get("daytona") if isinstance(meta, dict) else None
    if isinstance(daytona, dict) and daytona.get("reward") is not None:
        return float(daytona["reward"])
    direct = getattr(sample, "reward", None)
    if isinstance(direct, (int, float)):
        return float(direct)
    if isinstance(direct, dict):
        key = getattr(args, "reward_key", None)
        if key and key in direct:
            return float(direct[key])
    label = getattr(sample, "label", None)
    if isinstance(label, dict) and "reward" in label:
        return float(label["reward"])
    return 0.0


# Alias matching Slime docs naming.
custom_rm = reward
