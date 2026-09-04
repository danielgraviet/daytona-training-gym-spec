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


def correlation_attributes(
    *,
    run_id: str,
    rollout_id: str,
    project_id: str | None = None,
    sample_id: str | None = None,
    sandbox_id: str | None = None,
    worker_id: str | None = None,
    training_step: object | None = None,
    rollout_batch_id: str | None = None,
) -> dict[str, object]:
    return {
        "project_id": project_id or "",
        "run_id": run_id,
        "rollout_id": rollout_id,
        "sample_id": sample_id or "",
        "sandbox_id": sandbox_id or "",
        "worker_id": worker_id or "",
        "training_step": "" if training_step is None else training_step,
        "rollout_batch_id": rollout_batch_id or "",
    }
