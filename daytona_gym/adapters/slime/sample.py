from __future__ import annotations

from typing import Any

from daytona_gym.runtime.rollout import DaytonaTrajectory
from daytona_gym.runtime.types import Tokenizer, TrajectoryEvent


def prompt_text(sample: Any) -> str:
    prompt = getattr(sample, "prompt", "")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(prompt)


def set_sample_status(sample: Any, name: str) -> None:
    status_cls = getattr(type(sample), "Status", None)
    if status_cls is None:
        sample.status = name.lower()
        return
    sample.status = getattr(status_cls, name.upper())


def apply_trajectory(sample: Any, trajectory: DaytonaTrajectory, tokenizer: Tokenizer) -> None:
    existing_tokens = list(getattr(sample, "tokens", []) or [])
    prompt_tokens = existing_tokens if existing_tokens else tokenizer.encode(trajectory.prompt)

    response_text_parts: list[str] = []
    response_tokens: list[int] = []
    loss_mask: list[int] = []

    for event in trajectory.events:
        if event.type == "error":
            continue
        encoded = tokenizer.encode(event.text)
        response_text_parts.append(event.text)
        response_tokens.extend(encoded)
        trainable = 1 if event.type == "generation" else 0
        loss_mask.extend([trainable] * len(encoded))

    sample.tokens = prompt_tokens + response_tokens
    sample.response = "".join(response_text_parts)
    sample.response_length = len(response_tokens)
    sample.loss_mask = loss_mask
    sample.metadata.setdefault("daytona", {}).update(serializable_metadata(trajectory))
    set_sample_status(sample, trajectory.status.upper())


def serializable_metadata(trajectory: DaytonaTrajectory) -> dict[str, Any]:
    return {
        "run_id": trajectory.run_id,
        "rollout_id": trajectory.rollout_id,
        "sample_id": trajectory.sample_id,
        "sandbox_id": trajectory.sandbox_id,
        "status": trajectory.status,
        "error_code": trajectory.error_code,
        "error_message": trajectory.error_message,
        "final_response": trajectory.final_response,
        "started_at": trajectory.started_at.isoformat(),
        "finished_at": trajectory.finished_at.isoformat() if trajectory.finished_at else None,
        "events": [_event_summary(event) for event in trajectory.events],
    }


def _event_summary(event: TrajectoryEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "tool_name": event.tool_name,
        "ok": event.ok,
        "exit_code": event.exit_code,
        "error_code": event.error_code,
        "text_chars": len(event.text),
        "started_at": event.started_at.isoformat(),
        "finished_at": event.finished_at.isoformat(),
    }


def attach_cancelled_metadata(sample: Any, *, run_id: str, rollout_id: str) -> None:
    payload = sample.metadata.setdefault("daytona", {})
    payload.update(
        {
            "run_id": run_id,
            "rollout_id": rollout_id,
            "status": "aborted",
            "cancelled": True,
        }
    )
    set_sample_status(sample, "ABORTED")
