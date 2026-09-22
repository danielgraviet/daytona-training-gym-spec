from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from daytona_gym.adapters.slime import generate
from daytona_gym.cli import main as cli_main
from daytona_gym.runtime.fake import FakeEnvironmentRuntime
from daytona_gym.runtime.generation import ScriptedGenerator
from daytona_gym.telemetry import dashboard as dash_mod
from daytona_gym.telemetry import dashboard_data
from tests.helpers import FakeSlimeSample, final_turn, make_args, tool_turn


async def _write_jsonl(path: Path, *, n: int = 2) -> None:
    runtime = FakeEnvironmentRuntime()
    for index in range(n):
        generator = ScriptedGenerator(
            [
                tool_turn("run_tests", {"command": "pytest"}),
                final_turn("fixed"),
            ]
        )
        sample = FakeSlimeSample(prompt="fix", index=index)
        args = make_args(
            runtime=runtime,
            generator=generator,
            daytona_telemetry_path=str(path),
        )
        try:
            await generate(args, sample, {})
            args.daytona_telemetry_store.flush()
        finally:
            args.daytona_telemetry_store.close()


async def test_dashboard_data_summarizes_run(tmp_path: Path) -> None:
    path = tmp_path / "dogfood.jsonl"
    await _write_jsonl(path)
    listed = dashboard_data.list_run_files(tmp_path)
    assert len(listed) == 1
    assert listed[0]["stem"] == "dogfood"
    assert listed[0]["n_rollouts"] == 2
    assert listed[0]["mean_reward"] is not None

    detail = dashboard_data.run_detail(path)
    assert len(detail["rollouts"]) == 2
    rid = detail["rollouts"][0]["rollout_id"]
    rollout = dashboard_data.rollout_detail(path, rid)
    assert rollout["rollout_id"] == rid
    assert rollout["steps"]
    assert isinstance(rollout["wall_decomposition"], dict)


async def test_dashboard_http_pages(tmp_path: Path) -> None:
    path = tmp_path / "dogfood.jsonl"
    await _write_jsonl(path, n=1)
    runs_dir = tmp_path

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            body, content_type = dash_mod._route(unquote(urlparse(self.path).path), runs_dir)
            payload = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        time.sleep(0.05)
        conn = HTTPConnection("127.0.0.1", port, timeout=2)

        conn.request("GET", "/")
        res = conn.getresponse()
        body = res.read().decode()
        assert res.status == 200
        assert "dogfood.jsonl" in body

        conn.request("GET", "/run/dogfood")
        res = conn.getresponse()
        body = res.read().decode()
        assert res.status == 200
        assert "rollout_0" in body

        conn.request("GET", "/run/dogfood/rollout/rollout_0")
        res = conn.getresponse()
        body = res.read().decode()
        assert res.status == 200
        assert any(token in body for token in ("provision", "generate", "run_tests", "reward"))

        conn.request("GET", "/api/runs")
        res = conn.getresponse()
        payload = json.loads(res.read().decode())
        assert payload[0]["stem"] == "dogfood"
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_export_static_html(tmp_path: Path) -> None:
    path = tmp_path / "dogfood.jsonl"
    await _write_jsonl(path, n=1)
    out = tmp_path / "dashboard.html"
    written = dash_mod.export_static(tmp_path, out)
    text = written.read_text()
    assert "dogfood.jsonl" in text
    assert "rollout_0" in text


def test_cli_help_mentions_dash(capsys) -> None:
    assert cli_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "dg dash" in out
    assert "--share" in out


def test_detect_runpod_and_ssh(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    assert dash_mod.detect_serve_context().kind == "local"

    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 123 5.6.7.8 22")
    assert dash_mod.detect_serve_context().kind == "ssh"

    monkeypatch.setenv("RUNPOD_POD_ID", "abc123pod")
    ctx = dash_mod.detect_serve_context()
    assert ctx.kind == "runpod"
    assert ctx.runpod_pod_id == "abc123pod"


def test_runpod_access_hint(capsys, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "podxyz")
    ctx = dash_mod.detect_serve_context()
    dash_mod._print_access_hints(ctx, host="0.0.0.0", port=8765)
    out = capsys.readouterr().out
    assert "live public URL" in out


def test_parse_tunnel_url() -> None:
    from daytona_gym.telemetry.dashboard_tunnel import parse_public_url

    text = "INF | https://random-words-1234.trycloudflare.com"
    assert parse_public_url(text) == "https://random-words-1234.trycloudflare.com"


def test_parse_remote_port() -> None:
    from daytona_gym.telemetry.dashboard_sync import parse_remote_target

    assert parse_remote_target("root@1.2.3.4:12713") == ("root@1.2.3.4", 12713)
    assert parse_remote_target("root@1.2.3.4") == ("root@1.2.3.4", None)


def test_sync_runs_via_scp(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from daytona_gym.telemetry.dashboard_sync import sync_runs_from_ssh

    local_runs = tmp_path / "runs"
    remote_payload = tmp_path / "fake_remote_runs"
    remote_payload.mkdir()
    (remote_payload / "a.jsonl").write_text('{"ok":1}\n')

    def fake_run(args, capture_output=False, text=False):  # noqa: ANN001
        assert args[0] == "scp"
        dest = Path(args[-1])
        dest.mkdir(parents=True, exist_ok=True)
        for src in remote_payload.iterdir():
            (dest / src.name).write_text(src.read_text())
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sync_runs_from_ssh(
        target="root@64.247.201.60",
        local_runs=local_runs,
        ssh_port=12713,
        identity=Path("/tmp/fake_key"),
    )
    assert (local_runs / "a.jsonl").read_text() == '{"ok":1}\n'
