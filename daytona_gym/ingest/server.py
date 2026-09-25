"""``dg ingest`` — durable run history + dashboard in one small process.

Storage is the same ``runs/`` layout the local dashboard reads, so every
dashboard page / API works unchanged:

    <data>/<run_id>.jsonl           telemetry (appended by shippers)
    <data>/<run_id>.progress.json   latest progress snapshot
    <data>/<run_id>.ingest.json     source offset, generation, last_seen

Write API (Bearer token):

    POST /v1/runs/<id>/telemetry   X-DG-Offset / X-DG-Next / X-DG-Generation
    PUT  /v1/runs/<id>/progress
    POST /v1/runs/<id>/heartbeat

Reads (dashboard + ``/api/*``) need the same token: ``Authorization: Bearer``
for tools, or a cookie set by ``/login`` for browsers. Telemetry contains
prompts and model outputs, so nothing is public by default. Put this behind
HTTPS (a reverse proxy / platform TLS) in production.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from daytona_gym.ingest.config import INGEST_TOKEN_ENV, valid_run_id

MAX_BODY_BYTES = 8 * 1024 * 1024
_COOKIE = "dg_token"

_LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daytona Gym sign-in</title>
<style>body{background:#0c0f14;color:#e8eef7;font-family:system-ui,sans-serif;
display:grid;place-items:center;min-height:100vh;margin:0;padding:0 16px}
form{background:#141a22;border:1px solid #243041;border-radius:10px;padding:24px;
max-width:420px;width:100%}input{width:100%;box-sizing:border-box;padding:8px;
margin:12px 0;background:#0c0f14;color:#e8eef7;border:1px solid #243041;border-radius:6px}
button{padding:8px 14px;background:#3d9cf0;color:#fff;border:0;border-radius:6px}
p{color:#8b9bb0;font-size:14px}.bad{color:#f07178}</style></head><body>
<form method="post" action="/login"><h1 style="margin-top:0;font-size:20px">Daytona Gym</h1>
<p>Paste the ingest token (<code>DAYTONA_GYM_INGEST_TOKEN</code>).</p>__ERR__
<input type="password" name="token" autocomplete="current-password" autofocus>
<input type="hidden" name="next" value="__NEXT__">
<button type="submit">Open dashboard</button></form></body></html>"""


class IngestStore:
    """Append-only per-run files with idempotent, offset-checked writes."""

    def __init__(self, root: Path, *, clock=time.time) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, run_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(run_id, threading.Lock())

    def _meta_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.ingest.json"

    def read_meta(self, run_id: str) -> dict[str, Any]:
        try:
            data = json.loads(self._meta_path(run_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"source_offset": 0, "generation": 0}
        return data if isinstance(data, dict) else {"source_offset": 0, "generation": 0}

    def _write_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        _atomic_write(self._meta_path(run_id), json.dumps(meta).encode("utf-8"))

    def append(
        self, run_id: str, body: bytes, *, offset: int, next_offset: int, generation: int
    ) -> tuple[bool, dict[str, Any]]:
        """Returns (accepted, meta). Rejects (409) when offsets don't line up."""
        with self._lock(run_id):
            meta = self.read_meta(run_id)
            if generation > int(meta.get("generation", 0)):
                meta["generation"] = generation
                meta["source_offset"] = 0
            if generation < int(meta.get("generation", 0)) or offset != int(
                meta.get("source_offset", 0)
            ):
                return False, meta
            if body:
                with (self.root / f"{run_id}.jsonl").open("ab") as fh:
                    fh.write(body)
            meta["source_offset"] = next_offset
            meta["last_seen"] = self._clock()
            meta["bytes"] = int(meta.get("bytes", 0)) + len(body)
            self._write_meta(run_id, meta)
            return True, meta

    def put_progress(self, run_id: str, body: bytes) -> None:
        with self._lock(run_id):
            _atomic_write(self.root / f"{run_id}.progress.json", body)
            self.touch(run_id, locked=True)

    def touch(self, run_id: str, *, locked: bool = False) -> None:
        def _do() -> None:
            meta = self.read_meta(run_id)
            meta["last_seen"] = self._clock()
            self._write_meta(run_id, meta)

        if locked:
            _do()
        else:
            with self._lock(run_id):
                _do()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _validate_ndjson(body: bytes) -> str | None:
    if body and not body.endswith(b"\n"):
        return "telemetry body must end with a newline"
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "telemetry body must be JSON lines"
        if not isinstance(record, dict):
            return "each telemetry line must be a JSON object"
    return None


def make_handler(store: IngestStore, *, token: str | None):
    from daytona_gym.telemetry import dashboard as dash

    class Handler(BaseHTTPRequestHandler):
        server_version = "daytona-gym-ingest/0.1"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # ---- auth ------------------------------------------------------
        def _presented_token(self) -> str | None:
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                return auth[7:].strip()
            for part in (self.headers.get("Cookie") or "").split(";"):
                name, _, value = part.strip().partition("=")
                if name == _COOKIE:
                    return unquote(value)
            return None

        def _authorized(self) -> bool:
            if token is None:
                return True
            presented = self._presented_token()
            return presented is not None and hmac.compare_digest(presented, token)

        # ---- responses -------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _read_body(self) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._json(413, {"error": f"body must be 0..{MAX_BODY_BYTES} bytes"})
                return None
            return self.rfile.read(length) if length else b""

        # ---- GET: health, login, dashboard -----------------------------
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/healthz":
                self._json(200, {"ok": True})
                return
            if path == "/login":
                nxt = parse_qs(parsed.query).get("next", ["/"])[0]
                self._login_page(nxt)
                return
            if not self._authorized():
                if path.startswith("/api/"):
                    self._json(401, {"error": "unauthorized"})
                else:
                    self._send(302, b"", "text/plain", {"Location": f"/login?next={quote(path)}"})
                return
            if path == "/favicon.ico":
                self._send(204, b"", "text/plain")
                return
            try:
                body, ctype = dash._route(path, store.root)
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
                return
            self._send(200, body.encode("utf-8") if isinstance(body, str) else body, ctype)

        def _login_page(self, nxt: str, error: str = "") -> None:
            safe_next = nxt if nxt.startswith("/") and not nxt.startswith("//") else "/"
            html = _LOGIN_PAGE.replace("__NEXT__", quote(safe_next, safe="/")).replace(
                "__ERR__", f'<p class="bad">{error}</p>' if error else ""
            )
            self._send(401 if error else 200, html.encode("utf-8"), "text/html; charset=utf-8")

        # ---- POST/PUT: login + ingest ----------------------------------
        def do_POST(self) -> None:  # noqa: N802
            raw = urlparse(self.path).path
            if unquote(raw) == "/login":
                self._login_submit()
                return
            self._ingest("POST", raw)

        def do_PUT(self) -> None:  # noqa: N802
            self._ingest("PUT", urlparse(self.path).path)

        def _login_submit(self) -> None:
            body = self._read_body()
            if body is None:
                return
            form = parse_qs(body.decode("utf-8", errors="replace"))
            presented = form.get("token", [""])[0].strip()
            nxt = form.get("next", ["/"])[0]
            if token is not None and not hmac.compare_digest(presented, token):
                self._login_page(nxt, "That token was not accepted.")
                return
            secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
            cookie = f"{_COOKIE}={quote(presented)}; HttpOnly; SameSite=Strict; Path=/{secure}"
            location = nxt if nxt.startswith("/") and not nxt.startswith("//") else "/"
            self._send(303, b"", "text/plain", {"Location": location, "Set-Cookie": cookie})

        def _ingest(self, method: str, raw_path: str) -> None:
            # Split first, then decode: an encoded "/" stays inside one segment
            # and is rejected by the run-id check instead of changing the route.
            parts = [unquote(p) for p in raw_path.split("/") if p]
            if len(parts) != 4 or parts[:2] != ["v1", "runs"]:
                self._json(404, {"error": "not found"})
                return
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            run_id, what = parts[2], parts[3]
            if not valid_run_id(run_id):
                self._json(400, {"error": "invalid run id"})
                return
            body = self._read_body()
            if body is None:
                return
            if method == "POST" and what == "telemetry":
                problem = _validate_ndjson(body)
                if problem:
                    self._json(400, {"error": problem})
                    return
                try:
                    offset = int(self.headers.get("X-DG-Offset", ""))
                    nxt = int(self.headers.get("X-DG-Next", ""))
                    gen = int(self.headers.get("X-DG-Generation", "0"))
                except ValueError:
                    self._json(400, {"error": "X-DG-Offset / X-DG-Next must be integers"})
                    return
                if nxt < offset:
                    self._json(400, {"error": "X-DG-Next < X-DG-Offset"})
                    return
                ok, meta = store.append(run_id, body, offset=offset, next_offset=nxt, generation=gen)
                self._json(200 if ok else 409, {
                    "source_offset": meta.get("source_offset", 0),
                    "generation": meta.get("generation", 0),
                })
                return
            if method == "PUT" and what == "progress":
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                if not isinstance(parsed, dict):
                    self._json(400, {"error": "progress must be a JSON object"})
                    return
                store.put_progress(run_id, body)
                self._json(200, {"ok": True})
                return
            if method == "POST" and what == "heartbeat":
                store.touch(run_id)
                self._json(200, {"ok": True})
                return
            self._json(404, {"error": "not found"})

    return Handler


def serve(data_dir: Path, *, host: str, port: int, token: str | None) -> ThreadingHTTPServer:
    store = IngestStore(data_dir)
    return ThreadingHTTPServer((host, port), make_handler(store, token=token))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dg ingest",
        description="Durable run history: receive worker telemetry + serve the dashboard.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("gym-data"), help="Storage dir (default: ./gym-data)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 behind a proxy/platform)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable the token check (local testing only; telemetry is sensitive)",
    )
    args = parser.parse_args(argv)

    token = (os.environ.get(INGEST_TOKEN_ENV) or "").strip() or None
    if token is None and not args.no_auth:
        print(
            f"refusing to start without {INGEST_TOKEN_ENV} (or pass --no-auth for local testing)",
            file=sys.stderr,
        )
        return 2
    if token is not None and len(token) < 16:
        print(f"{INGEST_TOKEN_ENV} must be at least 16 characters", file=sys.stderr)
        return 2
    httpd = serve(args.data_dir, host=args.host, port=args.port, token=None if args.no_auth else token)
    print(f"dg ingest  data={args.data_dir.resolve()}  http://{args.host}:{httpd.server_address[1]}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
