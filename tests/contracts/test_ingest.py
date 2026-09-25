"""Phase 2: durable off-box run history (shipper → ``dg ingest`` → dashboard)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from daytona_gym.ingest.config import WORKER_LOST_AFTER_SECONDS, IngestConfig, valid_run_id
from daytona_gym.ingest.server import serve
from daytona_gym.ingest.shipper import Scrubber, TelemetryShipper
from daytona_gym.telemetry.progress import write_progress

TOKEN = "test-ingest-token-0123456789"
SECRET = "dtn_0123456789abcdef0123456789"


@pytest.fixture
def ingest(tmp_path: Path):
    data = tmp_path / "ingest-data"
    httpd = serve(data, host="127.0.0.1", port=0, token=TOKEN)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield url, data
    httpd.shutdown()
    httpd.server_close()


def _worker_files(tmp_path: Path, run_id: str = "run_ship") -> tuple[Path, Path]:
    runs = tmp_path / "worker" / "runs"
    runs.mkdir(parents=True)
    return runs / f"{run_id}.jsonl", runs


def _span(i: int, **attrs) -> str:
    return json.dumps(
        {
            "type": "span",
            "name": "rollout",
            "started_at": f"2026-09-25T00:00:{i:02d}+00:00",
            "duration_seconds": 1.0,
            "attributes": {"rollout_id": f"rollout_{i}", "training_step": 0, "status": "completed", **attrs},
        }
    )


def _shipper(url: str, path: Path, run_id: str = "run_ship", **kw) -> TelemetryShipper:
    return TelemetryShipper(
        IngestConfig(url=url, token=TOKEN),
        telemetry_path=path,
        run_id=run_id,
        scrubber=kw.pop("scrubber", Scrubber({})),
        **kw,
    )


def _drain(shipper: TelemetryShipper) -> None:
    for _ in range(50):
        if not shipper.ship_once():
            return
    raise AssertionError("shipper never caught up")


def _get(url: str, *, token: str | None = TOKEN, cookie: str | None = None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_ships_telemetry_and_progress_then_dashboard_serves_it(ingest, tmp_path) -> None:
    url, data = ingest
    path, runs = _worker_files(tmp_path)
    path.write_text("".join(_span(i) + "\n" for i in range(3)))
    write_progress(runs, "run_ship", phase="live", message="rolling", status="running")

    shipper = _shipper(url, path, max_chunk_bytes=300)  # force several chunks
    _drain(shipper)

    assert (data / "run_ship.jsonl").read_bytes() == path.read_bytes()
    assert json.loads((data / "run_ship.progress.json").read_text())["phase"] == "live"
    status, body = _get(f"{url}/api/runs/run_ship/analysis")
    assert status == 200 and json.loads(body)["totals"]["n_rollouts"] == 3

    # Appends keep flowing incrementally.
    with path.open("a") as fh:
        fh.write(_span(3) + "\n")
    _drain(shipper)
    assert (data / "run_ship.jsonl").read_bytes() == path.read_bytes()


def test_retries_are_idempotent_and_resync_after_restart(ingest, tmp_path) -> None:
    url, data = ingest
    path, _ = _worker_files(tmp_path)
    path.write_text("".join(_span(i) + "\n" for i in range(4)))
    _drain(_shipper(url, path))
    before = (data / "run_ship.jsonl").read_bytes()

    # A restarted worker process starts at offset 0: server answers 409 → resync.
    again = _shipper(url, path)
    _drain(again)
    assert (data / "run_ship.jsonl").read_bytes() == before


def test_truncated_file_starts_a_new_generation(ingest, tmp_path) -> None:
    url, data = ingest
    path, _ = _worker_files(tmp_path)
    path.write_text(_span(0) + "\n" + _span(1) + "\n")
    shipper = _shipper(url, path)
    _drain(shipper)
    path.write_text(_span(9) + "\n")  # rotated / truncated
    _drain(shipper)
    lines = (data / "run_ship.jsonl").read_text().splitlines()
    assert [json.loads(line)["attributes"]["rollout_id"] for line in lines] == [
        "rollout_0",
        "rollout_1",
        "rollout_9",
    ]


def test_oversized_line_is_skipped_not_stalled(ingest, tmp_path) -> None:
    url, data = ingest
    path, _ = _worker_files(tmp_path)
    path.write_text(_span(0, blob="x" * 5000) + "\n" + _span(1) + "\n")
    shipper = _shipper(url, path, max_chunk_bytes=1000)
    _drain(shipper)
    assert shipper.skipped_lines == 1
    kept = (data / "run_ship.jsonl").read_text().splitlines()
    assert [json.loads(line)["attributes"]["rollout_id"] for line in kept] == ["rollout_1"]


def test_secret_values_never_leave_the_box(ingest, tmp_path) -> None:
    url, data = ingest
    path, runs = _worker_files(tmp_path)
    path.write_text(_span(0, stdout=f"export DAYTONA_API_KEY={SECRET}; hf_abcdefghijklmnopqrstuvwxyz") + "\n")
    write_progress(runs, "run_ship", phase="live", message=f"using {SECRET}", status="running")
    scrub = Scrubber({"DAYTONA_API_KEY": SECRET, "HOME": "/root"})
    _drain(_shipper(url, path, scrubber=scrub))

    shipped = (data / "run_ship.jsonl").read_text() + (data / "run_ship.progress.json").read_text()
    assert SECRET not in shipped
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in shipped
    assert "***" in shipped
    json.loads((data / "run_ship.jsonl").read_text())  # still valid JSON


def test_auth_required_for_writes_and_reads(ingest, tmp_path) -> None:
    url, _ = ingest
    path, _ = _worker_files(tmp_path)
    path.write_text(_span(0) + "\n")
    bad = TelemetryShipper(
        IngestConfig(url=url, token="wrong-token-wrong-token"),
        telemetry_path=path,
        run_id="run_ship",
        scrubber=Scrubber({}),
    )
    with pytest.raises(PermissionError):
        bad.ship_once()

    assert _get(f"{url}/api/runs", token=None)[0] == 401
    assert _get(f"{url}/healthz", token=None)[0] == 200

    # Browser flow: /login sets an HttpOnly cookie that unlocks reads.
    req = urllib.request.Request(
        f"{url}/login", data=f"token={TOKEN}&next=/".encode(), method="POST"
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):  # noqa: ANN002, ANN003
            return None

    opener = urllib.request.build_opener(NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as caught:
        opener.open(req, timeout=5)
    assert caught.value.code == 303
    cookie = caught.value.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert _get(f"{url}/api/runs", token=None, cookie=cookie.split(";")[0])[0] == 200


def test_rejects_bad_run_ids_and_garbage(ingest) -> None:
    url, _ = ingest
    assert not valid_run_id("../etc/passwd") and not valid_run_id(".hidden")
    headers = {"Authorization": f"Bearer {TOKEN}", "X-DG-Offset": "0", "X-DG-Next": "4"}
    for path, body, code in [
        ("/v1/runs/..%2Fescape/telemetry", b"{}\n", 400),
        ("/v1/runs/run_ok/telemetry", b"not json\n", 400),
        ("/v1/runs/run_ok/telemetry", b"[1]\n", 400),
    ]:
        req = urllib.request.Request(url + path, data=body, method="POST", headers=headers)
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
        assert caught.value.code == code, path


def test_pod_dies_mid_run_history_survives_as_worker_lost(ingest, tmp_path) -> None:
    """Phase 2 exit criterion."""
    url, data = ingest
    path, runs = _worker_files(tmp_path)
    path.write_text("".join(_span(i) + "\n" for i in range(2)))
    write_progress(runs, "run_ship", phase="live", message="rolling", status="running")
    _drain(_shipper(url, path))
    # ... the pod is terminated here: no final status, no more heartbeats.
    meta_path = data / "run_ship.ingest.json"
    meta = json.loads(meta_path.read_text())
    meta["last_seen"] = time.time() - WORKER_LOST_AFTER_SECONDS - 1
    meta_path.write_text(json.dumps(meta))

    status, body = _get(f"{url}/api/runs/run_ship/live")
    live = json.loads(body)
    assert status == 200 and live["status"] == "worker_lost" and live["done"] is True
    runs_list = json.loads(_get(f"{url}/api/runs")[1])
    assert runs_list[0]["run_status"] == "worker_lost"
    analysis = json.loads(_get(f"{url}/api/runs/run_ship/analysis")[1])
    assert analysis["totals"]["n_rollouts"] == 2  # telemetry up to the loss


def test_terminal_status_is_not_overridden_by_silence(ingest, tmp_path) -> None:
    url, data = ingest
    path, runs = _worker_files(tmp_path)
    path.write_text(_span(0) + "\n")
    write_progress(runs, "run_ship", phase="completed", message="done", status="completed", done=True)
    _drain(_shipper(url, path))
    meta_path = data / "run_ship.ingest.json"
    meta = json.loads(meta_path.read_text())
    meta["last_seen"] = 0
    meta_path.write_text(json.dumps(meta))
    assert json.loads(_get(f"{url}/api/runs/run_ship/live")[1])["status"] == "completed"


def test_background_loop_ships_and_stop_drains(ingest, tmp_path) -> None:
    url, data = ingest
    path, runs = _worker_files(tmp_path)
    path.write_text(_span(0) + "\n")
    shipper = _shipper(url, path, interval_seconds=0.05).start()
    with path.open("a") as fh:
        fh.write(_span(1) + "\n")
    write_progress(runs, "run_ship", phase="completed", message="done", status="completed", done=True)
    shipper.stop()
    assert (data / "run_ship.jsonl").read_bytes() == path.read_bytes()
    assert json.loads((data / "run_ship.progress.json").read_text())["status"] == "completed"


def test_unreachable_ingest_backs_off_without_raising(tmp_path) -> None:
    path, _ = _worker_files(tmp_path)
    path.write_text(_span(0) + "\n")
    shipper = TelemetryShipper(
        IngestConfig(url="http://127.0.0.1:9", token=TOKEN),  # nothing listens on port 9
        telemetry_path=path,
        run_id="run_ship",
        interval_seconds=0.01,
        scrubber=Scrubber({}),
    ).start()
    time.sleep(0.3)
    shipper._stop.set()  # noqa: SLF001 — don't wait out the drain in tests
    assert shipper.errors >= 1 and shipper.bytes_shipped == 0


def test_open_prefers_ingest_url_without_local_server(tmp_path, monkeypatch) -> None:
    from daytona_gym.gym.run import TrainingRun
    from daytona_gym.gym.worker import _FORWARD_ENV_KEYS, _start_laptop_dash_for_run

    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", "https://gym.example.com/")
    monkeypatch.setenv("DAYTONA_GYM_INGEST_TOKEN", TOKEN)
    monkeypatch.setattr(
        "daytona_gym.telemetry.dashboard.start_dashboard",
        lambda **_: (_ for _ in ()).throw(AssertionError("no local dash with ingest")),
    )
    run = TrainingRun(run_id="r1", telemetry_path=str(tmp_path / "r1.jsonl"), command=[], env={}, runtime_env={})
    assert run.open() == "https://gym.example.com/run/r1"
    assert run._live_api_url() == "https://gym.example.com/api/runs/r1/live"
    url, handle = _start_laptop_dash_for_run(run_id="r1", host="h", remote_repo="/r", identity=None)
    assert url == "https://gym.example.com/run/r1" and handle is None
    assert {"DAYTONA_GYM_INGEST_URL", "DAYTONA_GYM_INGEST_TOKEN"} <= set(_FORWARD_ENV_KEYS)


def test_rm_deletes_a_run_with_auth_only(ingest, tmp_path, monkeypatch) -> None:
    from daytona_gym.ingest.server import rm_main

    url, data = ingest
    path, runs = _worker_files(tmp_path)
    path.write_text(_span(0) + "\n")
    write_progress(runs, "run_ship", phase="live", message="x", status="running")
    _drain(_shipper(url, path))
    assert (data / "run_ship.jsonl").exists()

    req = urllib.request.Request(f"{url}/v1/runs/run_ship", method="DELETE")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(req, timeout=5)  # noqa: S310
    assert caught.value.code == 401

    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", url)
    monkeypatch.setenv("DAYTONA_GYM_INGEST_TOKEN", TOKEN)
    assert rm_main(["run_ship", "--yes"]) == 0
    assert not any(data.glob("run_ship*"))
    assert rm_main(["run_ship", "--yes"]) == 1  # already gone → 404


def test_half_ingest_config_fails_loudly_and_off_is_announced(tmp_path, monkeypatch, capsys) -> None:
    """Regression: a pod with one of the two vars silently used a local tunnel."""
    from daytona_gym.ingest.shipper import start_shipper_from_env
    from daytona_gym.runtime.errors import DaytonaError

    path, _ = _worker_files(tmp_path)
    assert start_shipper_from_env(path, "run_ship") is None
    assert "ingest: OFF" in capsys.readouterr().out

    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", "https://gym.example.com")
    with pytest.raises(DaytonaError, match="DAYTONA_GYM_INGEST_TOKEN is not set"):
        start_shipper_from_env(path, "run_ship")


def test_push_backfills_finished_run_idempotently(ingest, tmp_path, capsys) -> None:
    from daytona_gym.ingest.shipper import push_files

    url, data = ingest
    path, runs = _worker_files(tmp_path, "run_backfill")
    path.write_text("".join(_span(i) + "\n" for i in range(5)))
    write_progress(runs, "run_backfill", phase="completed", message="done", status="completed", done=True)
    cfg = IngestConfig(url=url, token=TOKEN)

    assert push_files([path], config=cfg) == 0
    assert (data / "run_backfill.jsonl").read_bytes() == path.read_bytes()
    assert json.loads((data / "run_backfill.progress.json").read_text())["status"] == "completed"
    assert push_files([path], config=cfg) == 0  # second push: nothing new, no dupes
    assert (data / "run_backfill.jsonl").read_bytes() == path.read_bytes()
    assert "pushed run_backfill" in capsys.readouterr().out


def test_unreachable_ingest_warns_but_does_not_block_launch(tmp_path, monkeypatch, capsys) -> None:
    from daytona_gym.ingest.shipper import start_shipper_from_env

    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", "http://127.0.0.1:9")  # nothing listens
    monkeypatch.setenv("DAYTONA_GYM_INGEST_TOKEN", TOKEN)
    path, _ = _worker_files(tmp_path)
    shipper = start_shipper_from_env(path, "run_ship")
    assert shipper is not None  # still ships later
    shipper._stop.set()  # noqa: SLF001
    out = capsys.readouterr().out
    assert "is not reachable" in out and "dg ingest deploy --daytona" in out


def test_reachable_ingest_does_not_warn(ingest, tmp_path, monkeypatch, capsys) -> None:
    from daytona_gym.ingest.shipper import start_shipper_from_env

    url, _ = ingest
    monkeypatch.setenv("DAYTONA_GYM_INGEST_URL", url)
    monkeypatch.setenv("DAYTONA_GYM_INGEST_TOKEN", TOKEN)
    path, _ = _worker_files(tmp_path)
    start_shipper_from_env(path, "run_ship").stop(drain_timeout=1)
    assert "WARNING" not in capsys.readouterr().out
