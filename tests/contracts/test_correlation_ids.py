"""Every span emitted through the Slime adapter must carry step-level ids.

Without ``worker_id`` / ``training_step`` / ``rollout_batch_id`` no per-step
analytics are possible (they were empty in real H100/A100 runs).
"""

from __future__ import annotations

import asyncio

import pytest

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry import batches
from daytona_gym.telemetry.batches import RolloutBatchTracker
from daytona_gym.telemetry.store import InMemoryTelemetryStore
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture(autouse=True)
def _fresh_tracker(monkeypatch, clock) -> RolloutBatchTracker:
    tracker = RolloutBatchTracker(min_gap_seconds=1.0, clock=clock)
    monkeypatch.setattr(batches, "_DEFAULT", tracker)
    monkeypatch.delenv("DAYTONA_TRAINING_STEP", raising=False)
    monkeypatch.delenv("DAYTONA_WORKER_ID", raising=False)
    return tracker


def _args(store: InMemoryTelemetryStore, **overrides):
    runtime = FakeEnvironmentRuntime()
    generator = ScriptedGenerator(
        [tool_turn("run_command", {"command": "echo hi"}), final_turn("done")]
    )
    return make_args(
        runtime=runtime,
        generator=generator,
        daytona_telemetry_store=store,
        daytona_reward_function=lambda _a, _s: 1.0,
        **overrides,
    )


async def test_every_span_has_step_worker_and_batch_ids() -> None:
    store = InMemoryTelemetryStore()
    await generate(_args(store), FakeSlimeSample(prompt="p", index=0), {})

    spans = store.spans
    names = {s.name for s in spans}
    assert {"rollout", "sandbox.provision", "reward.compute", "rollout.outcome"} <= names
    for span in spans:
        attrs = span.attributes
        assert attrs["worker_id"], span.name
        assert attrs["training_step"] == 0, span.name
        assert attrs["rollout_batch_id"] == "step_0", span.name

    provision = next(s for s in spans if s.name == "sandbox.provision")
    assert provision.attributes["sandbox_id"]
    outcome = next(s for s in spans if s.name == "rollout.outcome")
    assert outcome.attributes["training_step_source"] == "derived"


async def test_explicit_step_and_worker_win(monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_WORKER_ID", "runpod:abc")
    store = InMemoryTelemetryStore()
    await generate(
        _args(store, daytona_training_step=7),
        FakeSlimeSample(prompt="p", index=1),
        {},
    )
    for span in store.spans:
        assert span.attributes["training_step"] == 7
        assert span.attributes["rollout_batch_id"] == "step_7"
        assert span.attributes["worker_id"] == "runpod:abc"
    outcome = next(s for s in store.spans if s.name == "rollout.outcome")
    assert outcome.attributes["training_step_source"] == "explicit"


async def test_sequential_batches_get_new_steps_concurrent_share_one(clock) -> None:
    store = InMemoryTelemetryStore()
    # Batch 0: two concurrent rollouts, plus a straggler submitted a moment after
    # the first two drained (sub-gap dip must NOT start a new step).
    await asyncio.gather(
        generate(_args(store), FakeSlimeSample(prompt="a", index=0), {}),
        generate(_args(store), FakeSlimeSample(prompt="b", index=1), {}),
    )
    clock.now += 0.05
    await generate(_args(store), FakeSlimeSample(prompt="b2", index=3), {})
    # Batch 1: after a trainer step (idle well past the gap).
    clock.now += 5.0
    await generate(_args(store), FakeSlimeSample(prompt="c", index=2), {})

    steps = {
        s.attributes["rollout_id"]: s.attributes["training_step"]
        for s in store.spans
        if s.name == "rollout"
    }
    assert steps == {"rollout_0": 0, "rollout_1": 0, "rollout_3": 0, "rollout_2": 1}


def test_tracker_releases_on_exception(clock) -> None:
    tracker = RolloutBatchTracker(min_gap_seconds=1.0, clock=clock)
    with pytest.raises(RuntimeError):
        with tracker.track() as step:
            assert step == 0
            raise RuntimeError("boom")
    clock.now += 2.0
    with tracker.track() as step:
        assert step == 1
