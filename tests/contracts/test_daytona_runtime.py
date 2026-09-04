from __future__ import annotations

from types import SimpleNamespace

from daytona_gym.adapters.slime import generate
from daytona_gym.runtime.daytona import DaytonaEnvironmentRuntime
from daytona_gym.runtime.errors import DaytonaError, ErrorCode
from daytona_gym.runtime.factory import build_environment_runtime
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.runtime.types import EnvironmentSpec, ToolAction, ToolName
from tests.fake_sdk import FakeAsyncDaytona
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


def _runtime(client: FakeAsyncDaytona | None = None, **kwargs: object) -> DaytonaEnvironmentRuntime:
    return DaytonaEnvironmentRuntime(client=client or FakeAsyncDaytona(), **kwargs)


async def test_create_from_snapshot_sets_labels() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    spec = EnvironmentSpec(
        snapshot="coding-env",
        metadata={"run_id": "run_1", "rollout_id": "rollout_9"},
    )

    handle = await runtime.create(spec)

    assert handle.sandbox_id == "sbx_1"
    assert handle.run_id == "run_1"
    params = client.created[0]["params"]
    assert params.snapshot == "coding-env"
    assert params.labels["run_id"] == "run_1"
    assert params.labels["rollout_id"] == "rollout_9"
    assert params.labels["component"] == "daytona-gym"
    assert params.ephemeral is True
    assert params.auto_stop_interval == 0
    await runtime.close(handle)
    assert client.deleted == ["sbx_1"]
    assert runtime.leaked_sandbox_ids == ()


async def test_create_from_image_when_no_snapshot() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    handle = await runtime.create(EnvironmentSpec(image="debian:12.9"))
    params = client.created[0]["params"]
    assert params.image == "debian:12.9"
    assert getattr(params, "snapshot", None) is None
    await runtime.close(handle)


async def test_provision_failure_is_typed() -> None:
    runtime = _runtime(FakeAsyncDaytona(fail_create=True))
    try:
        await runtime.create(EnvironmentSpec())
        raise AssertionError("expected provision failure")
    except DaytonaError as exc:
        assert exc.code is ErrorCode.SANDBOX_PROVISION_FAILED
        assert exc.details["cause"] == "RuntimeError"


async def test_create_timeout_is_typed() -> None:
    runtime = _runtime(FakeAsyncDaytona(fail_create_timeout=True))
    try:
        await runtime.create(EnvironmentSpec())
        raise AssertionError("expected timeout")
    except DaytonaError as exc:
        assert exc.code is ErrorCode.SANDBOX_TIMEOUT


async def test_run_command_and_files() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    env = await runtime.create(EnvironmentSpec(metadata={"run_id": "r", "rollout_id": "a"}))

    echo = await runtime.execute(
        env,
        ToolAction(name=ToolName.RUN_COMMAND, arguments={"command": "echo hello"}),
    )
    assert echo.ok is True
    assert echo.stdout == "hello\n"

    await runtime.execute(
        env,
        ToolAction(
            name=ToolName.WRITE_FILE,
            arguments={"path": "src/app.py", "content": "print(1)\n"},
        ),
    )
    read = await runtime.execute(
        env,
        ToolAction(name=ToolName.READ_FILE, arguments={"path": "src/app.py"}),
    )
    assert read.stdout == "print(1)\n"
    assert "src" in client.created[0]["sandbox"].fs.folders

    missing = await runtime.execute(
        env,
        ToolAction(name=ToolName.READ_FILE, arguments={"path": "nope.txt"}),
    )
    assert missing.ok is False
    await runtime.close(env)


async def test_apply_patch_uses_git_apply() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    env = await runtime.create(EnvironmentSpec())
    patch = "--- a/x\n+++ b/x\n"
    result = await runtime.execute(
        env,
        ToolAction(name=ToolName.APPLY_PATCH, arguments={"patch": patch}),
    )
    assert result.ok is True
    sandbox = client.created[0]["sandbox"]
    assert sandbox.files["/tmp/daytona-gym.patch"] == patch.encode()
    assert "git apply" in sandbox.process.commands[-1]["command"]
    await runtime.close(env)


async def test_run_tests_executes_command() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    env = await runtime.create(EnvironmentSpec())
    await runtime.execute(
        env,
        ToolAction(name=ToolName.RUN_TESTS, arguments={"command": "pytest -q"}),
    )
    command = client.created[0]["sandbox"].process.commands[-1]["command"]
    assert command == "pytest -q"
    await runtime.close(env)


async def test_tool_timeout_from_sdk() -> None:
    runtime = _runtime(FakeAsyncDaytona(hang_exec=True))
    env = await runtime.create(EnvironmentSpec())
    try:
        await runtime.execute(
            env,
            ToolAction(name=ToolName.RUN_COMMAND, arguments={"command": "echo hi"}),
        )
        raise AssertionError("expected tool timeout")
    except DaytonaError as exc:
        assert exc.code is ErrorCode.TOOL_TIMEOUT
    await runtime.close(env)
    assert runtime.leaked_sandbox_ids == ()


async def test_blocked_env_vars_are_not_sent_to_sandbox() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(
        client,
        env_vars={"DAYTONA_API_KEY": "secret", "LANG": "C"},
    )
    env = await runtime.create(EnvironmentSpec())
    await runtime.execute(
        env,
        ToolAction(
            name=ToolName.RUN_COMMAND,
            arguments={
                "command": "echo hi",
                "env": {"DAYTONA_API_KEY": "secret", "FOO": "bar"},
            },
        ),
    )
    create_params = client.created[0]["params"]
    assert create_params.env_vars == {"LANG": "C"}
    exec_env = client.created[0]["sandbox"].process.commands[-1]["env"]
    assert exec_env == {"FOO": "bar"}
    await runtime.close(env)


async def test_close_is_idempotent() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    env = await runtime.create(EnvironmentSpec())
    await runtime.close(env)
    await runtime.close(env)
    assert client.deleted == ["sbx_1"]


async def test_reset_issues_git_cleanup() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    env = await runtime.create(EnvironmentSpec())
    await runtime.reset(env)
    command = client.created[0]["sandbox"].process.commands[-1]["command"]
    assert "git reset --hard HEAD" in command
    await runtime.close(env)


async def test_path_traversal_rejected() -> None:
    runtime = _runtime()
    env = await runtime.create(EnvironmentSpec())
    try:
        await runtime.execute(
            env,
            ToolAction(name=ToolName.READ_FILE, arguments={"path": "../etc/passwd"}),
        )
        raise AssertionError("expected user code error")
    except DaytonaError as exc:
        assert exc.code is ErrorCode.USER_CODE_ERROR
    await runtime.close(env)


async def test_generate_uses_sdk_runtime_and_cleans_up() -> None:
    client = FakeAsyncDaytona()
    runtime = _runtime(client)
    generator = ScriptedGenerator(
        [
            tool_turn("write_file", {"path": "note.txt", "content": "alpha"}),
            tool_turn("read_file", {"path": "note.txt"}),
            final_turn("alpha"),
        ]
    )
    sample = FakeSlimeSample(prompt="task", index=4, rollout_id=4)
    args = make_args(runtime=runtime, generator=generator)

    await generate(args, sample, {})

    assert sample.status is FakeSlimeSample.Status.COMPLETED
    assert "alpha" in sample.response
    assert client.deleted == ["sbx_1"]
    assert runtime.leaked_sandbox_ids == ()


def test_factory_defaults_to_sdk_runtime() -> None:
    args = SimpleNamespace()
    runtime = build_environment_runtime(args)
    assert isinstance(runtime, DaytonaEnvironmentRuntime)
    assert args.daytona_environment_runtime is runtime


def test_factory_fake_flag() -> None:
    args = SimpleNamespace(daytona_use_fake_runtime=True)
    runtime = build_environment_runtime(args)
    assert isinstance(runtime, FakeEnvironmentRuntime)
