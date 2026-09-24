"""CPU tests for BYO Worker payload + LocalWorker wiring."""

from __future__ import annotations

from pathlib import Path

from daytona_gym import (
    LocalWorker,
    PromptJsonlDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    SshWorker,
    TrainConfig,
)
from daytona_gym.gym.remote_job import load_config
from daytona_gym.gym.worker import config_to_remote_payload


def _config(tmp_path: Path) -> TrainConfig:
    prompts = tmp_path / "examples" / "coding_dogfood" / "prompts"
    prompts.mkdir(parents=True)
    path = prompts / "coding_one.jsonl"
    path.write_text('{"prompt":"x","label":"1"}\n', encoding="utf-8")
    return TrainConfig(
        model=Qwen25_3B(
            slime_root=str(tmp_path / "slime"),
            megatron_root=str(tmp_path / "Megatron-LM"),
            hf_checkpoint=str(tmp_path / "hf"),
            ref_load=str(tmp_path / "ref"),
        ),
        dataset=PromptJsonlDataset(path),
        recipe=Qwen25_3B_Recipe(batch_size=1),
        repo=tmp_path,
        run_name="run_worker_test",
    )


def test_remote_payload_roundtrip(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    payload = config_to_remote_payload(
        cfg,
        remote_repo="/root/daytona-training-gym-spec",
        skip_preflight=True,
        preflight_timeout_seconds=30,
        open=True,
        open_browser=False,
    )
    assert payload["run_name"] == "run_worker_test"
    assert payload["dataset"]["path"].endswith(
        "examples/coding_dogfood/prompts/coding_one.jsonl"
    )
    assert payload["model"]["name"] == "Qwen2.5-3B-Instruct"
    assert payload["open"] is True

    # Paths on Mac won't exist for /root/... model dirs — only check construct
    remote_cfg = load_config(payload)
    assert remote_cfg.run_name == "run_worker_test"
    assert remote_cfg.model is not None
    assert remote_cfg.recipe.batch_size == 1


def test_local_worker_dry_run(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run = cfg.launch(worker=LocalWorker(), dry_run=True)
    assert run.dry_run is True
    assert run.training_run_id == "run_worker_test"


def test_ssh_worker_dry_run_no_network(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    worker = SshWorker(host="root@127.0.0.1", port=9, identity="/tmp/nope")
    run = cfg.launch(worker=worker, dry_run=True)
    assert run.dry_run is True


def test_ssh_worker_parses_markers(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from daytona_gym.gym.run import TrainingRun

    cfg = _config(tmp_path)
    worker = SshWorker(
        host="root@1.2.3.4",
        port=22,
        identity="/tmp/key",
        pull=False,
        transport="exec",
    )
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    def fake_run(cmd, capture_output=False, text=False):  # noqa: ANN001
        assert cmd[0] == "scp"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=False, bufsize=0):  # noqa: ANN001
            self.cmd = cmd
            self.stdout = iter(
                [
                    "noise\n",
                    "__DG_RUN_ID__=run_remote_1\n",
                    "__DG_TELEMETRY__=/root/runs/run_remote_1.jsonl\n",
                    "__DG_DASHBOARD__=https://x.trycloudflare.com/run/run_remote_1\n",
                    "__DG_RETURNCODE__=0\n",
                ]
            )

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    run = cfg.launch(worker=worker, dry_run=False, open=True)
    assert isinstance(run, TrainingRun)
    assert run.run_id == "run_remote_1"
    assert run.dashboard_url == "https://x.trycloudflare.com/run/run_remote_1"
    assert run.returncode == 0


def test_ssh_worker_parses_detached_markers(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from daytona_gym.gym.run import TrainingRun

    cfg = _config(tmp_path)
    worker = SshWorker(
        host="root@1.2.3.4",
        port=22,
        identity="/tmp/key",
        pull=False,
        transport="exec",
    )
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    def fake_run(cmd, capture_output=False, text=False):  # noqa: ANN001
        assert cmd[0] == "scp"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=False, bufsize=0):  # noqa: ANN001
            self.stdout = iter(
                [
                    "__DG_RUN_ID__=run_det_1\n",
                    "__DG_TELEMETRY__=/root/runs/run_det_1.jsonl\n",
                    "__DG_DASHBOARD__=https://x.trycloudflare.com/run/run_det_1\n",
                    "__DG_DETACHED__=1\n",
                    "__DG_RETURNCODE__=0\n",
                ]
            )

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    run = cfg.launch(worker=worker, dry_run=False, open=True, detach=True)
    assert isinstance(run, TrainingRun)
    assert run.detached is True
    assert run.returncode is None
    assert run.run_id == "run_det_1"
    assert run.dashboard_url == "https://x.trycloudflare.com/run/run_det_1"


def test_ensure_remote_repo_cmd_clones_when_missing() -> None:
    w = SshWorker(host="u@h", remote_repo="/root/daytona-training-gym-spec", pull=True)
    cmd = w._ensure_remote_repo_cmd()
    assert "git clone" in cmd
    assert "pip install -e" in cmd
    assert "/root/daytona-training-gym-spec" in cmd


def test_ssh_worker_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_GYM_SSH", "root@1.2.3.4")
    monkeypatch.setenv("DAYTONA_GYM_SSH_PORT", "2222")
    monkeypatch.setenv("DAYTONA_GYM_SSH_IDENTITY", "~/.ssh/id_ed25519")
    w = SshWorker.from_env()
    assert w.host == "root@1.2.3.4"
    assert w.port == 2222
