"""Contract: laptop detach with ingest set never starts PtyShell sync."""

from __future__ import annotations

from daytona_gym.gym import worker as worker_mod


def test_laptop_dash_uses_ingest_url(monkeypatch) -> None:
    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", "https://ingest.example")
    monkeypatch.setenv("DAYTONA_GYM_INGEST_TOKEN", "token-with-16chars")

    called = {"sync": False}

    def boom(*_a, **_k):
        called["sync"] = True
        raise AssertionError("must not PTY-sync when ingest is configured")

    monkeypatch.setattr(
        "daytona_gym.telemetry.dashboard_sync.sync_run_live_via_pty", boom
    )
    monkeypatch.setattr(
        "daytona_gym.telemetry.dashboard.start_dashboard", boom
    )

    url, handle = worker_mod._start_laptop_dash_for_run(
        run_id="run_abc",
        host="user@ssh.runpod.io",
        remote_repo="/root/daytona-training-gym-spec",
        identity=None,
    )
    assert url == "https://ingest.example/run/run_abc"
    assert handle is None
    assert called["sync"] is False
