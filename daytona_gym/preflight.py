"""Fast Daytona credential / API preflight before launching Slime.

Import / CLI:
  python -m daytona_gym.preflight
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.types import EnvironmentSpec


async def check_daytona(
    *,
    api_key: str | None = None,
    api_url: str | None = None,
    timeout_seconds: float = 60,
) -> None:
    """Create and destroy one ephemeral sandbox. Raises DaytonaError on failure."""
    key = api_key if api_key is not None else os.environ.get("DAYTONA_API_KEY")
    if not key:
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR,
            "DAYTONA_API_KEY is not set",
        )
    runtime = DaytonaEnvironmentRuntime(
        api_key=key,
        api_url=api_url if api_url is not None else os.environ.get("DAYTONA_API_URL"),
        ephemeral=True,
        create_timeout_seconds=timeout_seconds,
    )
    handle = None
    try:
        handle = await runtime.create(
            EnvironmentSpec(
                metadata={"run_id": "preflight", "rollout_id": "preflight"},
                timeout_seconds=timeout_seconds,
            )
        )
    finally:
        if handle is not None:
            try:
                await runtime.close(handle)
            except Exception:
                pass
        await runtime.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Daytona API credentials quickly")
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        asyncio.run(check_daytona(timeout_seconds=args.timeout_seconds))
    except DaytonaError as exc:
        print(f"daytona preflight FAILED [{exc.code}]: {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"  details: {exc.details}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"daytona preflight FAILED: {exc}", file=sys.stderr)
        return 1
    print("daytona preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
