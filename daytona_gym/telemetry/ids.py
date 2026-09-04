from __future__ import annotations

import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_run_id() -> str:
    return new_id("run")


def new_rollout_id() -> str:
    return new_id("rollout")


def new_sandbox_id() -> str:
    return new_id("sbx")
