from __future__ import annotations

from typing import Any

from daytona_gym.runtime.rollout import DaytonaTrajectory
from daytona_gym.runtime.types import OrdinalTokenizer, Tokenizer, TrajectoryEvent


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


def resolve_tokenizer(args: Any) -> Tokenizer:
    explicit = getattr(args, "daytona_tokenizer", None)
    if explicit is not None:
        return explicit
    slime_tok = _slime_tokenizer(args)
    if slime_tok is not None:
        return slime_tok
    return OrdinalTokenizer()


def _slime_tokenizer(args: Any) -> Tokenizer | None:
    try:
        from slime.rollout.sglang_rollout import GenerateState  # type: ignore
    except ImportError:
        return None
    try:
        state = GenerateState(args)
    except Exception:
        return None
    tokenizer = getattr(state, "tokenizer", None)
    if tokenizer is None:
        return None

    class _HFTokenizerAdapter:
        def encode(self, text: str) -> list[int]:
            # Prefer .encode(): calling the tokenizer may return a BatchEncoding
            # that is not a dict subclass (transformers 5+), and list(batch)
            # yields string keys — which later crashes slime tensorize.
            encode_fn = getattr(tokenizer, "encode", None)
            if callable(encode_fn):
                return [int(t) for t in encode_fn(text, add_special_tokens=False)]
            encoded = tokenizer(text, add_special_tokens=False)
            if isinstance(encoded, dict) or hasattr(encoded, "get"):
                return [int(t) for t in encoded["input_ids"]]
            if hasattr(encoded, "input_ids"):
                return [int(t) for t in encoded.input_ids]
            return [int(t) for t in encoded]

    return _HFTokenizerAdapter()


def set_sample_status(sample: Any, name: str) -> None:
    status_cls = getattr(type(sample), "Status", None)
    if status_cls is None:
        sample.status = name.lower()
        return
    sample.status = getattr(status_cls, name.upper())


def apply_trajectory(
    sample: Any,
    trajectory: DaytonaTrajectory,
    tokenizer: Tokenizer,
    *,
    args: Any | None = None,
) -> None:
    """Map a DaytonaTrajectory onto a Slime Sample (or duck-typed stand-in)."""
    if callable(getattr(sample, "append_response_tokens", None)):
        _apply_with_append(sample, trajectory, tokenizer, args=args)
    else:
        _apply_duck_typed(sample, trajectory, tokenizer)

    if getattr(sample, "reward", None) is None:
        inferred = infer_reward_from_trajectory(trajectory)
        if inferred is not None:
            sample.reward = inferred

    sample.metadata.setdefault("daytona", {}).update(serializable_metadata(trajectory))
    if getattr(sample, "reward", None) is not None:
        sample.metadata.setdefault("daytona", {})["reward"] = sample.reward
    set_sample_status(sample, trajectory.status.upper())


def _apply_with_append(
    sample: Any,
    trajectory: DaytonaTrajectory,
    tokenizer: Tokenizer,
    *,
    args: Any | None,
) -> None:
    raw_tokens = getattr(sample, "tokens", None)
    if isinstance(raw_tokens, str) or not raw_tokens:
        sample.tokens = [int(t) for t in tokenizer.encode(trajectory.prompt)]
    else:
        sample.tokens = [int(t) for t in raw_tokens]
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    if hasattr(sample, "rollout_log_probs"):
        sample.rollout_log_probs = None

    for event in trajectory.events:
        if event.type == "error":
            continue
        trainable = event.type == "generation"
        token_ids = _event_token_ids(event, tokenizer)
        log_probs = list(event.log_probs) if event.log_probs is not None else None
        if trainable and log_probs is not None and len(log_probs) != len(token_ids):
            log_probs = None
        sample.append_response_tokens(
            args,
            tokens=token_ids,
            log_probs=log_probs if trainable else None,
            trainable=trainable,
            text=event.text,
            update_terminal_info=False,
        )


def _apply_duck_typed(
    sample: Any,
    trajectory: DaytonaTrajectory,
    tokenizer: Tokenizer,
) -> None:
    existing_tokens = list(getattr(sample, "tokens", []) or [])
    prompt_tokens = existing_tokens if existing_tokens else tokenizer.encode(trajectory.prompt)

    response_text_parts: list[str] = []
    response_tokens: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    have_log_probs = False

    for event in trajectory.events:
        if event.type == "error":
            continue
        encoded = _event_token_ids(event, tokenizer)
        response_text_parts.append(event.text)
        response_tokens.extend(encoded)
        trainable = 1 if event.type == "generation" else 0
        loss_mask.extend([trainable] * len(encoded))
        if event.type == "generation" and event.log_probs is not None:
            have_log_probs = True
            if len(event.log_probs) == len(encoded):
                rollout_log_probs.extend(event.log_probs)
            else:
                rollout_log_probs.extend([0.0] * len(encoded))
        else:
            rollout_log_probs.extend([0.0] * len(encoded))

    sample.tokens = prompt_tokens + response_tokens
    sample.response = "".join(response_text_parts)
    sample.response_length = len(response_tokens)
    sample.loss_mask = loss_mask
    if have_log_probs:
        sample.rollout_log_probs = rollout_log_probs


def _event_token_ids(event: TrajectoryEvent, tokenizer: Tokenizer) -> list[int]:
    if event.token_ids is not None:
        return list(event.token_ids)
    return tokenizer.encode(event.text)


def infer_reward_from_trajectory(trajectory: DaytonaTrajectory) -> float | None:
    """Prefer last successful run_tests; otherwise leave reward unset."""
    last_tests: TrajectoryEvent | None = None
    for event in trajectory.events:
        if event.type == "tool" and event.tool_name == "run_tests":
            last_tests = event
    if last_tests is None:
        return None
    if last_tests.ok and (last_tests.exit_code or 0) == 0:
        return 1.0
    return 0.0


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
    from daytona_gym.runtime.security import clip_text

    preview, truncated = clip_text(event.text, 400)
    return {
        "type": event.type,
        "tool_name": event.tool_name,
        "ok": event.ok,
        "exit_code": event.exit_code,
        "error_code": event.error_code,
        "text_chars": len(event.text),
        "text_preview": preview,
        "text_preview_truncated": truncated,
        "token_count": len(event.token_ids) if event.token_ids is not None else None,
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
