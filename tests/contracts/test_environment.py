from __future__ import annotations

from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.types import EnvironmentSpec, ToolAction, ToolName


async def test_fake_runtime_isolates_filesystems() -> None:
    runtime = FakeEnvironmentRuntime()
    env_a = await runtime.create(EnvironmentSpec(metadata={"run_id": "r", "rollout_id": "a"}))
    env_b = await runtime.create(EnvironmentSpec(metadata={"run_id": "r", "rollout_id": "b"}))

    await runtime.execute(
        env_a,
        ToolAction(name=ToolName.WRITE_FILE, arguments={"path": "x.txt", "content": "A"}),
    )
    await runtime.execute(
        env_b,
        ToolAction(name=ToolName.WRITE_FILE, arguments={"path": "x.txt", "content": "B"}),
    )

    read_a = await runtime.execute(
        env_a, ToolAction(name=ToolName.READ_FILE, arguments={"path": "x.txt"})
    )
    read_b = await runtime.execute(
        env_b, ToolAction(name=ToolName.READ_FILE, arguments={"path": "x.txt"})
    )
    assert read_a.stdout == "A"
    assert read_b.stdout == "B"

    await runtime.close(env_a)
    await runtime.close(env_b)
    assert runtime.leaked_sandbox_ids == ()


async def test_reset_clears_files() -> None:
    runtime = FakeEnvironmentRuntime()
    env = await runtime.create(EnvironmentSpec())
    await runtime.execute(
        env,
        ToolAction(name=ToolName.WRITE_FILE, arguments={"path": "x.txt", "content": "keep"}),
    )
    await runtime.reset(env)
    result = await runtime.execute(
        env, ToolAction(name=ToolName.READ_FILE, arguments={"path": "x.txt"})
    )
    assert result.ok is False
    await runtime.close(env)


async def test_create_failure_is_typed() -> None:
    runtime = FakeEnvironmentRuntime(fail_create=True)
    try:
        await runtime.create(EnvironmentSpec())
        raise AssertionError("expected provision failure")
    except DaytonaError as exc:
        assert exc.code is ErrorCode.SANDBOX_PROVISION_FAILED
        assert exc.to_dict()["code"] == "sandbox_provision_failed"


async def test_close_is_idempotent() -> None:
    runtime = FakeEnvironmentRuntime()
    env = await runtime.create(EnvironmentSpec())
    await runtime.close(env)
    await runtime.close(env)
    assert runtime.closed_ids == (env.sandbox_id,)
