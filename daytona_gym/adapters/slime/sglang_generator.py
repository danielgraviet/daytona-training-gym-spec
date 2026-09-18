from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.generation import GenerationResult

PostFn = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class SupportsSGLangRouter(Protocol):
    sglang_router_ip: str
    sglang_router_port: int | str


class SGLangRouterGenerator:
    """GenerationBackend that calls Slime's SGLang router `/generate`.

    Mirrors slime examples/search-r1: POST to
    ``http://{sglang_router_ip}:{sglang_router_port}/generate``.
    """

    def __init__(
        self,
        args: Any,
        *,
        post_fn: PostFn | None = None,
        return_logprob: bool | None = None,
    ) -> None:
        self._args = args
        self._post_fn = post_fn or default_post
        if return_logprob is None:
            return_logprob = bool(getattr(args, "daytona_return_logprob", True))
        self._return_logprob = return_logprob

    @classmethod
    def from_args(cls, args: Any, *, post_fn: PostFn | None = None) -> SGLangRouterGenerator:
        if post_fn is None:
            post_fn = getattr(args, "daytona_sglang_post_fn", None)
        return cls(args, post_fn=post_fn)

    @property
    def router_url(self) -> str:
        ip = getattr(self._args, "sglang_router_ip", None)
        port = getattr(self._args, "sglang_router_port", None)
        if not ip or port is None:
            raise DaytonaError(
                ErrorCode.PLATFORM_ERROR,
                "sglang router not configured; set args.sglang_router_ip and "
                "args.sglang_router_port (Slime sets these when SGLang starts)",
            )
        return f"http://{ip}:{port}/generate"

    async def generate(self, conversation: str, sampling_params: dict) -> GenerationResult:
        payload: dict[str, Any] = {
            "text": conversation,
            "sampling_params": dict(sampling_params or {}),
        }
        if self._return_logprob:
            payload["return_logprob"] = True
        try:
            output = await self._post_fn(self.router_url, payload)
        except DaytonaError:
            raise
        except Exception as exc:
            raise DaytonaError(
                ErrorCode.INFERENCE_FAILED,
                f"sglang router request failed: {exc}",
            ) from exc
        return _result_from_sglang_output(output, return_logprob=self._return_logprob)


def _result_from_sglang_output(
    output: dict[str, Any],
    *,
    return_logprob: bool,
) -> GenerationResult:
    if not isinstance(output, dict):
        raise DaytonaError(
            ErrorCode.INFERENCE_FAILED,
            f"sglang returned non-object response: {type(output)!r}",
        )
    meta = output.get("meta_info") or {}
    finish = (meta.get("finish_reason") or {}) if isinstance(meta, dict) else {}
    finish_type = finish.get("type") if isinstance(finish, dict) else None
    if finish_type == "abort":
        raise DaytonaError(ErrorCode.INFERENCE_FAILED, "sglang generation aborted")

    text = str(output.get("text") or "")
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    if return_logprob:
        raw = meta.get("output_token_logprobs") if isinstance(meta, dict) else None
        if raw is None:
            raise DaytonaError(
                ErrorCode.INFERENCE_FAILED,
                "output_token_logprobs missing; ensure return_logprob=True on the payload",
            )
        try:
            token_ids = [int(item[1]) for item in raw]
            log_probs = [float(item[0]) for item in raw]
        except (TypeError, ValueError, IndexError) as exc:
            raise DaytonaError(
                ErrorCode.INFERENCE_FAILED,
                f"malformed output_token_logprobs: {exc}",
            ) from exc
        if len(token_ids) != len(log_probs):
            raise DaytonaError(
                ErrorCode.INFERENCE_FAILED,
                "token/logprob length mismatch from sglang",
            )

    request_id = None
    if isinstance(meta, dict):
        for key in ("id", "request_id", "model_request_id"):
            if meta.get(key) is not None:
                request_id = str(meta[key])
                break

    return GenerationResult(
        text=text,
        token_ids=token_ids,
        log_probs=log_probs,
        request_id=request_id,
    )


async def default_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Prefer Slime's HTTP helper; fall back to stdlib urllib."""
    try:
        from slime.utils.http_utils import post as slime_post  # type: ignore

        return await slime_post(url, payload)
    except ImportError:
        pass
    return await _urllib_post(url, payload)


async def _urllib_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    import asyncio
    import json
    import urllib.error
    import urllib.request

    def _sync() -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc
        return json.loads(body)

    return await asyncio.to_thread(_sync)
