from __future__ import annotations

from types import SimpleNamespace

import pytest

from daytona_gym.adapters.slime.generate import generate
from daytona_gym.adapters.slime.sglang_generator import (
    SGLangRouterGenerator,
    _result_from_sglang_output,
)
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import GenerationResult
from tests.helpers import FakeSlimeSample, final_turn, make_args


async def test_sglang_generator_parses_logprobs() -> None:
    async def fake_post(url: str, payload: dict) -> dict:
        assert url.endswith("/generate")
        assert payload["return_logprob"] is True
        assert payload["text"] == "hello"
        return {
            "text": '{"type":"final","content":"ok"}',
            "meta_info": {
                "id": "req-1",
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 11], [-0.2, 22], [-0.3, 33]],
            },
        }

    args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000)
    generator = SGLangRouterGenerator(args, post_fn=fake_post)
    result = await generator.generate("hello", {"temperature": 0.0})
    assert result.request_id == "req-1"
    assert result.token_ids == [11, 22, 33]
    assert result.log_probs == [-0.1, -0.2, -0.3]
    assert "final" in result.text


async def test_sglang_generator_maps_abort() -> None:
    with pytest.raises(DaytonaError) as exc_info:
        _result_from_sglang_output(
            {"text": "", "meta_info": {"finish_reason": {"type": "abort"}}},
            return_logprob=False,
        )
    assert exc_info.value.code is ErrorCode.INFERENCE_FAILED


async def test_sglang_generator_requires_router_args() -> None:
    generator = SGLangRouterGenerator(SimpleNamespace())
    with pytest.raises(DaytonaError) as exc_info:
        await generator.generate("x", {})
    assert exc_info.value.code is ErrorCode.PLATFORM_ERROR


async def test_generate_defaults_to_sglang_backend() -> None:
    calls: list[str] = []

    async def fake_post(url: str, payload: dict) -> dict:
        assert "10.0.0.1:19000" in url
        calls.append(payload["text"])
        return {
            "text": final_turn("done"),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.01, ord(c)] for c in final_turn("done")],
            },
        }

    runtime = FakeEnvironmentRuntime()
    args = make_args(runtime=runtime, generator=None)
    args.daytona_generator = None
    args.sglang_router_ip = "10.0.0.1"
    args.sglang_router_port = 19000
    args.daytona_sglang_post_fn = fake_post

    sample = FakeSlimeSample(prompt="fix it", index=1)
    await generate(args, sample, {"temperature": 0.0})

    assert isinstance(args.daytona_generator, SGLangRouterGenerator)
    assert sample.status is FakeSlimeSample.Status.COMPLETED
    assert sample.rollout_log_probs is not None
    assert len(sample.rollout_log_probs) == sample.response_length
    assert sample.loss_mask is not None
    assert all(mask == 1 for mask in sample.loss_mask)
    assert calls and calls[0].startswith("fix it")


async def test_result_from_output_without_logprob() -> None:
    result = _result_from_sglang_output(
        {
            "text": "hi",
            "meta_info": {"finish_reason": {"type": "stop"}},
        },
        return_logprob=False,
    )
    assert result == GenerationResult(text="hi", token_ids=None, log_probs=None, request_id=None)
