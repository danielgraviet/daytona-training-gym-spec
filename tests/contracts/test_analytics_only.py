"""Phase 5: analytics-only path (no TrainConfig) — ship CLI + env wiring."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from daytona_gym.adapters.slime.generate import _apply_analytics_env
from daytona_gym.cli import main as dg_main
from daytona_gym.ingest.server import serve
from daytona_gym.ingest.shipper import ship_main
from types import SimpleNamespace

TOKEN = "phase5-analytics-tok"  # ≥16 chars


def test_apply_analytics_env_from_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYTONA_TELEMETRY_PATH", "runs/ao.jsonl")
    monkeypatch.setenv("DAYTONA_RUN_ID", "run_analytics_demo")
    monkeypatch.setenv("DAYTONA_PROJECT_ID", "ao")
    args = SimpleNamespace()
    _apply_analytics_env(args)
    assert args.daytona_telemetry_path == "runs/ao.jsonl"
    assert args.daytona_run_id == "run_analytics_demo"
    assert args.daytona_project_id == "ao"


def test_ship_main_requires_ingest_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.delenv("DAYTONA_GYM_INGEST_URL", raising=False)
    monkeypatch.delenv("DAYTONA_GYM_INGEST_TOKEN", raising=False)
    path = tmp_path / "run_x.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert ship_main([str(path)]) == 2
    err = capsys.readouterr().err
    assert "DAYTONA_GYM_INGEST_URL" in err


def test_ship_main_pushes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "ingest"
    data.mkdir()
    httpd = serve(data, host="127.0.0.1", port=0, token=TOKEN)
    host, port = httpd.server_address[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://{host}:{port}"
    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", url)
    monkeypatch.setenv("DAYTONA_GYM_INGEST_TOKEN", TOKEN)

    path = tmp_path / "run_phase5.jsonl"
    path.write_text(
        '{"type":"span","name":"rollout","attributes":{"rollout_id":"r0"}}\n',
        encoding="utf-8",
    )
    try:
        assert ship_main([str(path)]) == 0
        stored = data / "run_phase5.jsonl"
        assert stored.is_file()
        assert "rollout" in stored.read_text(encoding="utf-8")
    finally:
        httpd.shutdown()


def test_dg_help_mentions_ship(capsys) -> None:
    assert dg_main(["-h"]) == 0
    out = capsys.readouterr().out
    assert "dg ship" in out
