"""Local dashboard over ``runs/*.jsonl`` (Svelte SPA + JSON/SSE APIs).

  dg dash                 # localhost:3000; auto-sync RunPod from .env
  dg open
  dg dash --port 3000
  dg dash --share         # optional Cloudflare quick tunnel
  dg dash --remote USER@HOST   # escape hatch (non-RunPod SSH)
  python -m daytona_gym.telemetry.dashboard --runs-dir runs --port 3000
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from daytona_gym.telemetry import dashboard_data as data
from daytona_gym.telemetry import progress as run_progress
from daytona_gym.telemetry.dashboard_sync import (
    resolve_auto_remote,
    resolve_remote_target,
    sync_runs_from_ssh,
)
from daytona_gym.telemetry.dashboard_tunnel import QuickTunnel


@dataclass
class DashboardHandle:
    """Background dashboard server (+ optional public tunnel)."""

    url: str
    local_url: str
    runs_dir: Path
    shared: bool
    _httpd: ThreadingHTTPServer = field(repr=False)
    _thread: threading.Thread = field(repr=False)
    _tunnel: QuickTunnel | None = field(default=None, repr=False)

    def stop(self) -> None:
        if self._tunnel is not None:
            self._tunnel.stop()
            self._tunnel = None
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        try:
            self._httpd.server_close()
        except Exception:
            pass
        if self._thread.is_alive() or self._thread.ident is not None:
            try:
                self._thread.join(timeout=5)
            except RuntimeError:
                pass


def start_dashboard(
    *,
    runs_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    share: bool | None = None,
    quiet: bool = False,
) -> DashboardHandle:
    """Start the dashboard in a background thread; return a live URL handle."""
    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    context = detect_serve_context()
    if share is None:
        # Prefer loopback. Cloudflare is opt-in via --share / share=True.
        share = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            if quiet:
                return
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            # SSE: keep one connection open; server pushes progress (no page reload).
            parts = [p for p in path.split("/") if p]
            if (
                len(parts) == 4
                and parts[0] == "api"
                and parts[1] == "runs"
                and parts[3] == "events"
            ):
                self._stream_run_events(parts[2])
                return
            try:
                body, content_type = _route(path, runs_dir)
            except FileNotFoundError as exc:
                self.send_error(404, str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, str(exc))
                return
            payload = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream_run_events(self, stem: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last = ""
            try:
                while True:
                    snap = _run_live_snapshot(runs_dir, stem)
                    payload = json.dumps(snap, separators=(",", ":"))
                    if payload != last:
                        self.wfile.write(
                            f"event: progress\ndata: {payload}\n\n".encode("utf-8")
                        )
                        self.wfile.flush()
                        last = payload
                    if snap.get("ready"):
                        self.wfile.write(b"event: ready\ndata: {}\n\n")
                        self.wfile.flush()
                        break
                    if snap.get("failed") or snap.get("done"):
                        evt = "failed" if snap.get("failed") else "done"
                        self.wfile.write(
                            f"event: {evt}\ndata: {payload}\n\n".encode("utf-8")
                        )
                        self.wfile.flush()
                        break
                    time.sleep(2.0)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    bind_host = "127.0.0.1" if share else host
    try:
        httpd = ThreadingHTTPServer((bind_host, port), Handler)
    except OSError:
        # Port busy (e.g. prior dg dash) — pick an ephemeral port.
        httpd = ThreadingHTTPServer((bind_host, 0), Handler)
    bound_port = httpd.server_address[1]
    local_url = f"http://127.0.0.1:{bound_port}/"
    if not quiet:
        print(f"Daytona Gym dashboard  {local_url}")
        print(f"runs dir: {runs_dir}")

    tunnel: QuickTunnel | None = None
    public_url: str | None = None
    if share:
        if not quiet:
            print("starting public tunnel (outbound; no RunPod port edits) …")
        tunnel = QuickTunnel(local_url.rstrip("/"))
        try:
            public_url = tunnel.start()
        except Exception as exc:  # noqa: BLE001
            print(f"tunnel failed: {exc}", file=sys.stderr)
            print(
                "falling back to local-only. Last-resort offline dump: "
                "dg dash --export runs/dashboard.html",
                file=sys.stderr,
            )
            tunnel = None
            public_url = None
        if public_url and not quiet:
            print()
            print("Open on your laptop (live):")
            print(f"  {public_url}")
            print("  (public while this process runs — treat traces as sensitive)")
            print()
    elif not quiet:
        _print_access_hints(context, host=bind_host, port=bound_port)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = (public_url or local_url).rstrip("/") + "/"
    should_open = open_browser and context.kind == "local" and not share
    if should_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return DashboardHandle(
        url=url,
        local_url=local_url,
        runs_dir=runs_dir,
        shared=bool(public_url),
        _httpd=httpd,
        _thread=thread,
        _tunnel=tunnel,
    )


def serve(
    *,
    runs_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    share: bool | None = None,
) -> None:
    handle = start_dashboard(
        runs_dir=runs_dir,
        host=host,
        port=port,
        open_browser=open_browser,
        share=share,
        quiet=False,
    )
    print("Ctrl+C to stop")
    try:
        while handle._thread.is_alive():
            handle._thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        handle.stop()


class ServeContext:
    __slots__ = ("kind", "runpod_pod_id")

    def __init__(self, kind: str, runpod_pod_id: str | None = None) -> None:
        self.kind = kind  # local | ssh | runpod
        self.runpod_pod_id = runpod_pod_id


def detect_serve_context() -> ServeContext:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id and Path("/etc/rp_environment").is_file():
        pod_id = _read_runpod_id("/etc/rp_environment")
    if pod_id:
        return ServeContext("runpod", runpod_pod_id=pod_id)
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        return ServeContext("ssh")
    return ServeContext("local")


def _read_runpod_id(path: str | Path) -> str | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if "POD_ID" in line.upper() and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _print_access_hints(context: ServeContext, *, host: str, port: int) -> None:
    print()
    print(f"Open locally:  http://127.0.0.1:{port}/")
    if context.kind in {"runpod", "ssh"}:
        print("On a GPU box, prefer pulling to your laptop (no Cloudflare):")
        print(f"  dg dash --remote user@ssh.runpod.io -i ~/.ssh/id_ed25519 --port {port}")
        print("Optional public tunnel only if you need it:  dg dash --share")
    print()


_STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json",
}


def spa_available() -> bool:
    return (_STATIC_DIR / "index.html").is_file()


def _spa_file(rel: str) -> tuple[bytes, str] | None:
    """Serve a file under dashboard_static/, or None if missing."""
    rel = rel.lstrip("/")
    if not rel or ".." in rel.split("/"):
        return None
    path = (_STATIC_DIR / rel).resolve()
    try:
        path.relative_to(_STATIC_DIR.resolve())
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path.read_bytes(), _MIME.get(path.suffix.lower(), "application/octet-stream")


def _spa_index() -> tuple[bytes, str]:
    raw = _spa_file("index.html")
    if raw is None:
        raise FileNotFoundError(
            "SPA index missing — run npm run build in dashboard_frontend/"
        )
    return raw


def export_static(runs_dir: Path, out_path: Path) -> Path:
    """Write one self-contained HTML file (no server needed on the laptop)."""
    runs_dir = Path(runs_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs = data.list_run_files(runs_dir)
    sections: list[str] = []
    for run in runs:
        if run.get("error"):
            sections.append(
                f"<section><h2>{_esc(run.get('name'))}</h2>"
                f"<p class='bad'>{_esc(run['error'])}</p></section>"
            )
            continue
        try:
            detail = data.run_detail(Path(run["path"]))
        except Exception as exc:  # noqa: BLE001
            sections.append(
                f"<section><h2>{_esc(run.get('name'))}</h2>"
                f"<p class='bad'>{_esc(exc)}</p></section>"
            )
            continue
        summary = detail.get("summary") or {}
        mean = summary.get("mean_reward")
        mean_s = f"{mean:.3f}" if isinstance(mean, float) else "-"
        rows = []
        for r in detail.get("rollouts") or []:
            rid = r["rollout_id"]
            try:
                rd = data.rollout_detail(Path(run["path"]), rid)
            except Exception:  # noqa: BLE001
                rd = {}
            step_lines = "".join(
                f"<li>{_esc(s.get('short') or s.get('name'))} "
                f"({_esc(_fmt_secs(s.get('duration_seconds')))})</li>"
                for s in (rd.get("steps") or [])[:40]
            )
            rows.append(
                "<tr>"
                f"<td>{_esc(rid)}</td>"
                f"<td>{_esc(r.get('status'))}</td>"
                f"<td>{_esc(r.get('reward'))}</td>"
                f"<td>{_esc(_fmt_secs(r.get('wall_seconds')))}</td>"
                f"<td><ul>{step_lines or '<li class=empty>no steps</li>'}</ul></td>"
                "</tr>"
            )
        sections.append(
            f"<section id='{_esc(run['stem'])}'>"
            f"<h2>{_esc(detail['name'])}</h2>"
            f"<p class='meta'>mean reward <strong>{_esc(mean_s)}</strong> · "
            f"n <strong>{len(detail.get('rollouts') or [])}</strong></p>"
            "<table><thead><tr>"
            "<th>rollout</th><th>status</th><th>reward</th><th>wall</th><th>timeline</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5 class=empty>none</td></tr>'}</tbody>"
            "</table></section>"
        )
    body = (
        "<nav class='crumb'>exported dashboard · open this file in a browser</nav>"
        + (
            "".join(sections)
            if sections
            else "<p class='empty'>No runs/*.jsonl found.</p>"
        )
    )
    out_path.write_text(_layout("Daytona Gym (export)", body), encoding="utf-8")
    return out_path


def _route(path: str, runs_dir: Path) -> tuple[str | bytes, str]:
    if path == "/api/overview" or path == "/api/runs" or path.startswith("/api/runs/"):
        return _api(path, runs_dir), "application/json; charset=utf-8"

    use_spa = spa_available()
    if use_spa:
        if path.startswith("/assets/"):
            asset = _spa_file(path)
            if asset is not None:
                return asset
            raise FileNotFoundError(path)
        if path in {"/", "/index.html"} or path.startswith("/run/"):
            return _spa_index()
        asset = _spa_file(path)
        if asset is not None:
            return asset

    # Legacy HTML fallback when SPA has not been built.
    if path in {"/", "/index.html"}:
        return _page_index(runs_dir), "text/html; charset=utf-8"
    if path.startswith("/run/"):
        parts = [p for p in path.split("/") if p]
        if len(parts) == 2:
            return _page_run(runs_dir, parts[1]), "text/html; charset=utf-8"
        if len(parts) == 4 and parts[2] == "rollout":
            return (
                _page_rollout(runs_dir, parts[1], parts[3]),
                "text/html; charset=utf-8",
            )
    raise FileNotFoundError(path)


def _api(path: str, runs_dir: Path) -> str:
    parts = [p for p in path.split("/") if p]
    if parts == ["api", "runs"]:
        return json.dumps(data.list_run_files(runs_dir))
    if parts == ["api", "overview"]:
        return json.dumps(_overview_snapshot(runs_dir))
    if len(parts) == 4 and parts[3] == "live":
        return json.dumps(_run_live_snapshot(runs_dir, parts[2]))
    if len(parts) == 4 and parts[3] == "analysis":
        return json.dumps(data.run_analysis(_resolve_run(runs_dir, parts[2])))
    if len(parts) == 4 and parts[3] == "charts":
        path_file = _resolve_run(runs_dir, parts[2])
        return json.dumps(data.run_charts(path_file))
    if len(parts) == 3:
        path_file = _resolve_run(runs_dir, parts[2])
        return json.dumps(data.run_detail(path_file))
    if len(parts) == 5 and parts[3] == "rollouts":
        path_file = _resolve_run(runs_dir, parts[2])
        return json.dumps(data.rollout_detail(path_file, parts[4]))
    raise FileNotFoundError(path)


def _overview_snapshot(runs_dir: Path) -> dict[str, Any]:
    """Compact index fingerprint so the browser can reload when anything changes."""
    runs = []
    for item in data.list_run_files(runs_dir):
        runs.append(
            {
                "stem": item.get("stem") or item.get("name"),
                "n": item.get("n_rollouts", 0),
                "statuses": item.get("statuses"),
                "run_status": item.get("run_status"),
                "phase": item.get("phase"),
                "mean_reward": item.get("mean_reward"),
                "mtime": item.get("mtime"),
                "progress_updated_at": item.get("progress_updated_at"),
            }
        )
    progress = []
    for stem in run_progress.list_progress_stems(runs_dir):
        prog = run_progress.read_progress(runs_dir, stem) or {}
        progress.append(
            {
                "stem": stem,
                "phase": prog.get("phase"),
                "message": prog.get("message"),
                "updated_at": prog.get("updated_at"),
                "status": prog.get("status"),
            }
        )
    return {"runs": runs, "progress": progress}


def _resolve_run(runs_dir: Path, stem: str) -> Path:
    root = Path(runs_dir).resolve()
    for candidate in (root / f"{stem}.jsonl", root / stem):  # stem or full filename
        resolved = candidate.resolve()
        # Served from a public ingest host too: never escape the runs dir.
        if resolved.parent == root and resolved.suffix == ".jsonl" and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"run not found: {stem}")


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_secs(value: object) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if v < 10:
        return f"{v:.2f}s"
    return f"{v:.1f}s"


def _layout(title: str, body: str, *, extra_head: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  {extra_head}
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --ink: #1a1a1a;
      --muted: #5c5c5c;
      --line: #d9d2c5;
      --card: #fffdf8;
      --accent: #0b6e4f;
      --bad: #8b1e1e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    header {{
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--line);
      background: var(--card);
    }}
    header h1 {{
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: -0.02em;
    }}
    header p {{ margin: 0.25rem 0 0; color: var(--muted); font-size: 0.95rem; }}
    main {{ padding: 1.25rem 1.5rem 3rem; max-width: 960px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
    }}
    th, td {{
      text-align: left;
      padding: 0.55rem 0.7rem;
      border-bottom: 1px solid var(--line);
      font-family: "IBM Plex Mono", "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.82rem;
    }}
    th {{
      font-family: inherit;
      font-size: 0.8rem;
      color: var(--muted);
      font-weight: 600;
    }}
    .meta {{
      display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem;
      margin: 0 0 1rem; color: var(--muted); font-size: 0.95rem;
    }}
    .meta strong {{ color: var(--ink); font-weight: 600; }}
    .ok {{ color: var(--accent); }}
    .bad {{ color: var(--bad); }}
    .empty {{ color: var(--muted); padding: 2rem 0; }}
    nav.crumb {{ margin-bottom: 1rem; font-size: 0.9rem; color: var(--muted); }}
    .bar {{
      display: flex; height: 0.65rem; background: var(--line); border-radius: 2px;
      overflow: hidden; margin: 0.35rem 0 1rem;
    }}
    .bar span {{ display: block; height: 100%; }}
    .b-inf {{ background: #2c5aa0; }}
    .b-sbx {{ background: #0b6e4f; }}
    .b-env {{ background: #a67c00; }}
    .b-oth {{ background: #7a7a7a; }}
    .phases {{ list-style: none; padding: 0; margin: 1rem 0; }}
    .phases li {{
      padding: 0.55rem 0.75rem;
      border: 1px solid var(--line);
      border-radius: 4px;
      margin: 0.35rem 0;
      background: var(--card);
    }}
    .phases li.active {{ border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }}
    .phases li.done {{ color: var(--muted); }}
    .phases li.failed {{ border-color: var(--bad); box-shadow: inset 3px 0 0 var(--bad); }}
    .phases li.active .phase-msg {{
      display: block;
      margin-top: 0.35rem;
      font-size: 0.85rem;
      color: var(--muted);
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
    }}
    .phases .tag {{
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 0.75rem;
      color: var(--accent);
      margin-right: 0.5rem;
    }}
    .phases li.failed .tag {{ color: var(--bad); }}
    .alive {{
      display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.75rem 1.25rem;
      margin: 0 0 1rem; padding: 0.85rem 1rem;
      background: var(--card); border: 1px solid var(--line); border-radius: 4px;
    }}
    .alive .clock {{
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 1.6rem; font-weight: 600; color: var(--ink);
      min-width: 4.5rem;
    }}
    .alive .clock.pulse {{ animation: dg-pulse 1.2s ease-in-out infinite; }}
    @keyframes dg-pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.55; }}
    }}
    .conn {{
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 0.75rem;
    }}
    .conn.ok {{ color: var(--accent); }}
    .conn.wait {{ color: var(--muted); }}
    .conn.bad {{ color: var(--bad); }}
    .activity {{
      list-style: none; padding: 0; margin: 0.75rem 0 1rem;
      max-height: 14rem; overflow: auto;
      border: 1px solid var(--line); border-radius: 4px; background: var(--card);
    }}
    .activity li {{
      padding: 0.4rem 0.75rem;
      border-bottom: 1px solid var(--line);
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .activity li:last-child {{ border-bottom: 0; color: var(--ink); }}
    .activity .at {{ color: var(--muted); margin-right: 0.5rem; }}
    pre.preview, pre.log-tail {{
      margin: 0.2rem 0 0.6rem 0;
      white-space: pre-wrap;
      color: var(--muted);
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 0.78rem;
    }}
    pre.log-tail {{
      max-height: 12rem;
      overflow: auto;
      padding: 0.65rem 0.75rem;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 4px;
    }}
    pre.log-tail:empty, pre.log-tail[hidden] {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>Daytona Training Gym</h1>
    <p>Local rollout dashboard · reads JSONL under <code>runs/</code></p>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
"""


def _page_index(runs_dir: Path) -> str:
    runs = data.list_run_files(runs_dir)
    progress_only = [
        stem
        for stem in run_progress.list_progress_stems(runs_dir)
        if not (runs_dir / f"{stem}.jsonl").is_file()
    ]
    if not runs and not progress_only:
        body = (
            f'<p class="empty">No <code>*.jsonl</code> files in '
            f"<code>{_esc(runs_dir)}</code>. Run a dogfood job first.</p>"
        )
        return _layout("Runs", body)
    rows = []
    for stem in progress_only:
        prog = run_progress.read_progress(runs_dir, stem) or {}
        phase = str(prog.get("phase") or "starting")
        message = str(prog.get("message") or "")
        short = message[:100] + ("…" if len(message) > 100 else "")
        if phase == "failed":
            status_cell = (
                f"<td class='bad'>failed"
                + (f" · {_esc(short)}" if short else "")
                + "</td>"
            )
        else:
            status_cell = (
                f"<td class='ok'>{_esc(phase)}"
                + (f" · {_esc(short)}" if short else "")
                + "</td>"
            )
        rows.append(
            "<tr>"
            f"<td><a href='/run/{_esc(stem)}'>{_esc(stem)}</a></td>"
            "<td>0</td>"
            f"{status_cell}"
            "<td>-</td><td>-</td>"
            "</tr>"
        )
    for run in runs:
        if run.get("error"):
            rows.append(
                f"<tr><td>{_esc(run['name'])}</td><td colspan='4' class='bad'>"
                f"{_esc(run['error'])}</td></tr>"
            )
            continue
        mean = run.get("mean_reward")
        mean_s = f"{mean:.3f}" if isinstance(mean, float) else "-"
        statuses = run.get("statuses") or {}
        status_s = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
        rows.append(
            "<tr>"
            f"<td><a href='/run/{_esc(run['stem'])}'>{_esc(run['name'])}</a></td>"
            f"<td>{run.get('n_rollouts', 0)}</td>"
            f"<td>{_esc(status_s)}</td>"
            f"<td>{_esc(mean_s)}</td>"
            f"<td>{_esc(_fmt_secs(run.get('wall_p50')))}</td>"
            "</tr>"
        )
    # Auto-reload when progress or JSONL set changes (SSE through tunnels can drop).
    script = """
    <script>
    (function () {
      let last = '';
      async function tick() {
        try {
          const res = await fetch('/api/overview', { cache: 'no-store' });
          if (!res.ok) return;
          const text = await res.text();
          if (last && text !== last) location.reload();
          last = text;
        } catch (e) {}
      }
      tick();
      setInterval(tick, 2000);
    })();
    </script>
    """
    body = f"""
    <nav class="crumb">runs</nav>
    <table>
      <thead><tr>
        <th>run</th><th>n</th><th>status</th><th>mean reward</th><th>wall p50</th>
      </tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    {script}
    """
    return _layout("Runs", body)


def _run_live_snapshot(runs_dir: Path, stem: str) -> dict:
    prog = run_progress.read_progress(runs_dir, stem) or {}
    n_rollouts = 0
    ready = False
    path_file: Path | None = None
    steps_seen: list[str] = []
    num_steps: int | None = None
    try:
        path_file = _resolve_run(runs_dir, stem)
        detail = data.run_detail(path_file)
        n_rollouts = len(detail.get("rollouts") or [])
        ready = n_rollouts > 0
        steps_seen = list(detail.get("steps_seen") or [])
        num_steps = detail.get("num_steps")
    except FileNotFoundError:
        pass
    phase = str(prog.get("phase") or ("live" if ready else "starting"))
    message = str(
        prog.get("message")
        or (
            f"{n_rollouts} rollout(s) ready"
            if ready
            else "Waiting for worker phases (dataset → model → Ray/Slime → sandboxes)…"
        )
    )
    failed = phase == "failed" or prog.get("status") == "failed"
    status = str(prog.get("status") or "")
    if not status:
        if failed:
            status = "failed"
        elif phase == "completed":
            status = "completed"
        elif ready and not prog and path_file is not None:
            status = data.infer_status_without_progress(path_file, has_rollouts=True)
            if status == "stale":
                message = "No progress file and telemetry idle ≥10m — run likely ended"
        elif ready:
            status = "running"
        else:
            status = "running" if phase else "pending"
    lived = data.apply_liveness(runs_dir, stem, status)
    if lived == "worker_lost" and status != "worker_lost":
        status = "worker_lost"
        message = "Worker stopped reporting (pod gone?) — showing telemetry up to that point"
        failed = True
    done = bool(
        prog.get("done")
        or status in {"completed", "failed", "worker_lost"}
        or phase in {"completed", "failed"}
    )
    # Only promote phase to "live" while still actively running. The worker's
    # progress message stops at "Initializing Megatron…" once Slime is up, so
    # describe training from the telemetry itself.
    detail_msg = prog.get("detail")
    if ready and not failed and not done and status == "running":
        phase = "live"
        detail_msg = message
        message = _training_message(n_rollouts, steps_seen, num_steps)
    # Freeze elapsed for terminal runs (UI must not keep ticking).
    elapsed_s = prog.get("elapsed_s")
    if done and not isinstance(elapsed_s, (int, float)):
        # Best-effort: activity first→last timestamps.
        activity = prog.get("activity") if isinstance(prog.get("activity"), list) else []
        times = [
            float(a["t"])
            for a in activity
            if isinstance(a, dict) and isinstance(a.get("t"), (int, float))
        ]
        if len(times) >= 2:
            elapsed_s = max(0.0, times[-1] - times[0])
        elif isinstance(prog.get("updated_at"), (int, float)) and times:
            elapsed_s = max(0.0, float(prog["updated_at"]) - times[0])
    log_tail = prog.get("log_tail") or []
    if not isinstance(log_tail, list):
        log_tail = []
    started_at = prog.get("started_at")
    if started_at is None and isinstance(prog.get("activity"), list) and prog["activity"]:
        first = prog["activity"][0]
        if isinstance(first, dict) and first.get("t") is not None:
            started_at = first["t"]
    return {
        "phase": phase,
        "message": message,
        "detail": detail_msg,
        "steps_seen": len(steps_seen),
        "num_steps": num_steps,
        "elapsed_s": elapsed_s,
        "started_at": started_at,
        "log_tail": [str(x) for x in log_tail[-12:]],
        "activity": prog.get("activity") if isinstance(prog.get("activity"), list) else [],
        "n_rollouts": n_rollouts,
        "ready": ready,
        "failed": failed,
        "done": done,
        "status": status,
        "returncode": prog.get("returncode"),
        "updated_at": prog.get("updated_at"),
    }


def _training_message(n_rollouts: int, steps_seen: list[str], num_steps: int | None) -> str:
    n = len(steps_seen)
    if num_steps and n >= num_steps:
        return (
            f"All {num_steps} steps' rollouts done ({n_rollouts} rollouts) — "
            "trainer finishing the last step / cleanup"
        )
    if num_steps and n:
        return f"Training — step {n}/{num_steps} ({n_rollouts} rollouts so far)"
    if n:
        return f"Training — {n} step(s) seen, {n_rollouts} rollouts so far"
    return f"Training — {n_rollouts} rollouts so far"


_PHASE_STEPS = (
    ("starting", "Run created"),
    ("model_download", "Download model weights"),
    ("model_convert", "Convert HF → Megatron"),
    ("ray_start", "Start Ray / Slime"),
    ("training", "Training — waiting for first rollout"),
    ("live", "Rollouts live"),
    ("completed", "Training complete"),
)


def _phase_items_html(phase: str) -> str:
    items = []
    seen_active = False
    failed = phase == "failed"
    for key, label in _PHASE_STEPS:
        if failed:
            # Mark everything up to training as muted; show failure banner separately.
            cls = "done" if key != "live" else ""
            tag = "ok" if cls == "done" else ""
        elif key == phase:
            cls = "active"
            seen_active = True
            tag = "now"
        elif key == "live" and phase == "live":
            cls = "active"
            seen_active = True
            tag = "now"
        elif not seen_active and key != "live":
            cls = "done"
            tag = "ok"
        else:
            cls = ""
            tag = ""
        if phase == "live" and key != "live":
            cls = "done"
            tag = "ok"
        items.append(
            f'<li data-phase="{_esc(key)}" class="{cls}">'
            + (f'<span class="tag">{tag}</span>' if tag else "")
            + f"{_esc(label)}</li>"
        )
    if failed:
        items.append(
            '<li data-phase="failed" class="failed">'
            '<span class="tag">err</span>Failed</li>'
        )
    return "".join(items)


def _page_run_pending(runs_dir: Path, stem: str) -> str:
    snap = _run_live_snapshot(runs_dir, stem)
    phase = str(snap.get("phase") or "starting")
    message = str(snap.get("message") or "Worker is starting…")
    failed = bool(snap.get("failed"))
    status_label = "failed" if failed else str(phase)
    status_cls = "bad" if failed else ""
    log_tail = snap.get("log_tail") or []
    log_text = "\n".join(str(x) for x in log_tail)
    activity = snap.get("activity") or []
    activity_html = "".join(
        "<li><span class='at'>"
        + _esc(time.strftime("%H:%M:%S", time.localtime(float(a.get("t") or 0))))
        + "</span>"
        + _esc(str(a.get("message") or ""))
        + "</li>"
        for a in activity
        if isinstance(a, dict)
    )
    elapsed0 = int(snap.get("elapsed_s") or 0)
    steps_json = json.dumps([[k, lab] for k, lab in _PHASE_STEPS])
    stem_json = json.dumps(stem)
    script = f"""
    <script>
    (function () {{
      const stem = {stem_json};
      const steps = {steps_json};
      const statusEl = document.getElementById('dg-status');
      const msgEl = document.getElementById('dg-message');
      const detailEl = document.getElementById('dg-detail');
      const logEl = document.getElementById('dg-log');
      const list = document.getElementById('dg-phases');
      const clockEl = document.getElementById('dg-clock');
      const connEl = document.getElementById('dg-conn');
      const actEl = document.getElementById('dg-activity');
      const ageEl = document.getElementById('dg-age');
      let stopped = false;
      let currentPhase = {json.dumps(phase)};
      let currentMsg = {json.dumps(message)};
      let serverElapsed = {elapsed0};
      let phaseLocalStart = Date.now() - ({elapsed0} * 1000);
      let lastServerAt = Date.now();
      let pollOk = false;

      function fmt(sec) {{
        sec = Math.max(0, Math.floor(sec));
        if (sec < 60) return sec + 's';
        const m = Math.floor(sec / 60), s = sec % 60;
        return m + 'm ' + String(s).padStart(2, '0') + 's';
      }}

      function renderPhases(phase, msg) {{
        let seen = false;
        const failed = phase === 'failed';
        let html = steps.map(([key, label]) => {{
          let cls = '';
          if (failed) {{
            cls = key !== 'live' ? 'done' : '';
          }} else if (phase === 'live') {{
            cls = key === 'live' ? 'active' : 'done';
          }} else if (key === phase) {{
            cls = 'active'; seen = true;
          }} else if (!seen && key !== 'live') {{
            cls = 'done';
          }}
          const tag = cls === 'active' ? 'now' : (cls === 'done' ? 'ok' : '');
          let extra = '';
          if (cls === 'active' && msg) {{
            extra = '<span class="phase-msg">' + msg.replace(/</g,'&lt;') + '</span>';
          }}
          return '<li data-phase="' + key + '" class="' + cls + '">'
            + (tag ? '<span class="tag">' + tag + '</span>' : '')
            + label + extra + '</li>';
        }}).join('');
        if (failed) {{
          html += '<li data-phase="failed" class="failed">'
            + '<span class="tag">err</span>Failed'
            + (msg ? '<span class="phase-msg">' + msg.replace(/</g,'&lt;') + '</span>' : '')
            + '</li>';
        }}
        list.innerHTML = html;
      }}

      function renderActivity(items) {{
        if (!actEl || !items || !items.length) return;
        actEl.innerHTML = items.map(a => {{
          const t = a.t ? new Date(a.t * 1000) : null;
          const hh = t ? t.toLocaleTimeString() : '';
          return '<li><span class="at">' + hh + '</span>'
            + String(a.message || '').replace(/</g,'&lt;') + '</li>';
        }}).join('');
        actEl.scrollTop = actEl.scrollHeight;
      }}

      function onProgress(data) {{
        lastServerAt = Date.now();
        pollOk = true;
        if (connEl) {{
          connEl.textContent = 'live';
          connEl.className = 'conn ok';
        }}
        if (data.message) {{
          currentMsg = data.message;
          msgEl.textContent = data.message;
        }}
        if (typeof data.elapsed_s === 'number') {{
          serverElapsed = data.elapsed_s;
          phaseLocalStart = Date.now() - (serverElapsed * 1000);
        }}
        if (detailEl) {{
          if (data.detail) {{
            detailEl.textContent = data.detail;
            detailEl.hidden = false;
          }}
        }}
        if (logEl) {{
          const tail = data.log_tail || [];
          if (tail.length) {{
            logEl.textContent = tail.join('\\n');
            logEl.hidden = false;
            logEl.scrollTop = logEl.scrollHeight;
          }}
        }}
        if (data.activity) renderActivity(data.activity);
        if (data.failed) {{
          statusEl.textContent = 'failed';
          statusEl.className = 'bad';
          currentPhase = 'failed';
          renderPhases('failed', currentMsg);
          if (clockEl) clockEl.classList.remove('pulse');
          return;
        }}
        if (data.phase) {{
          if (data.phase !== currentPhase) {{
            currentPhase = data.phase;
            if (typeof data.elapsed_s !== 'number') {{
              phaseLocalStart = Date.now();
              serverElapsed = 0;
            }}
          }}
          statusEl.textContent = data.ready ? 'live' : data.phase;
          statusEl.className = '';
          renderPhases(data.phase, currentMsg);
        }} else {{
          renderPhases(currentPhase, currentMsg);
        }}
      }}

      function tickClock() {{
        if (stopped && currentPhase === 'failed') return;
        const local = (Date.now() - phaseLocalStart) / 1000;
        const show = Math.max(serverElapsed || 0, local);
        if (clockEl) clockEl.textContent = fmt(show);
        if (ageEl) {{
          const lag = Math.floor((Date.now() - lastServerAt) / 1000);
          ageEl.textContent = pollOk
            ? ('updated ' + (lag < 2 ? 'just now' : lag + 's ago'))
            : 'waiting for first update…';
        }}
        if (connEl && pollOk) {{
          const lag = (Date.now() - lastServerAt) / 1000;
          if (lag > 12) {{
            connEl.textContent = 'stale — retrying';
            connEl.className = 'conn wait';
          }}
        }}
      }}

      async function poll() {{
        if (stopped) return;
        try {{
          const res = await fetch(
            '/api/runs/' + encodeURIComponent(stem) + '/live?t=' + Date.now(),
            {{ cache: 'no-store' }}
          );
          if (!res.ok) {{
            if (connEl) {{ connEl.textContent = 'http ' + res.status; connEl.className = 'conn bad'; }}
            return;
          }}
          const data = await res.json();
          onProgress(data);
          if (data.ready) {{
            stopped = true;
            location.reload();
            return;
          }}
          if (data.failed) {{
            stopped = true;
            if (clockEl) clockEl.classList.remove('pulse');
          }}
        }} catch (e) {{
          if (connEl) {{ connEl.textContent = 'reconnect…'; connEl.className = 'conn wait'; }}
        }}
      }}

      // Polling is primary — Cloudflare quick tunnels often drop EventSource.
      setInterval(poll, 1500);
      setInterval(tickClock, 250);
      poll();
      tickClock();
      renderPhases(currentPhase, currentMsg);

      try {{
        const es = new EventSource('/api/runs/' + encodeURIComponent(stem) + '/events');
        es.addEventListener('progress', (ev) => {{
          try {{ onProgress(JSON.parse(ev.data)); }} catch (e) {{}}
        }});
        es.addEventListener('ready', () => {{ stopped = true; es.close(); location.reload(); }});
        es.addEventListener('failed', (ev) => {{
          try {{ onProgress(Object.assign({{failed: true, phase: 'failed'}}, JSON.parse(ev.data))); }}
          catch (e) {{ onProgress({{failed: true, phase: 'failed'}}); }}
          stopped = true; es.close();
        }});
        es.onerror = () => {{ /* poll keeps us alive */ }};
      }} catch (e) {{}}
    }})();
    </script>
    """
    clock_cls = "clock" if failed else "clock pulse"
    body = f"""
    <nav class="crumb"><a href="/">runs</a> / {_esc(stem)}</nav>
    <div class="alive">
      <span id="dg-clock" class="{clock_cls}">{_esc(f"{elapsed0}s")}</span>
      <div>
        <div class="meta" style="margin:0">
          <span>status <strong id="dg-status" class="{status_cls}">{_esc(status_label)}</strong></span>
          <span id="dg-conn" class="conn wait">connecting…</span>
          <span id="dg-age" class="conn wait"></span>
        </div>
        <div id="dg-message" style="margin-top:0.35rem">{_esc(message)}</div>
        <div id="dg-detail" class="meta" style="margin:0.35rem 0 0" {'hidden' if not snap.get('detail') else ''}>{_esc(str(snap.get('detail') or ''))}</div>
      </div>
    </div>
    <p>This page polls every 1.5s (SSE is best-effort through the tunnel). The clock
    always ticks so a long Ray/Slime boot does not look frozen. Rollout timelines appear
    once training emits telemetry.</p>
    <ul class="phases" id="dg-phases">{_phase_items_html(phase)}</ul>
    <h2 style="font-size:0.95rem;margin:1rem 0 0.35rem;color:var(--muted)">Activity</h2>
    <ul class="activity" id="dg-activity">{activity_html or "<li>Waiting for worker heartbeats…</li>"}</ul>
    <h2 style="font-size:0.95rem;margin:1rem 0 0.35rem;color:var(--muted)">Worker log (tail)</h2>
    <pre class="log-tail" id="dg-log" {'hidden' if not log_text else ''}>{_esc(log_text)}</pre>
    {script}
    """
    return _layout(f"{stem} ({'failed' if failed else 'starting'})", body)


def _page_run(runs_dir: Path, stem: str) -> str:
    try:
        path_file = _resolve_run(runs_dir, stem)
    except FileNotFoundError:
        return _page_run_pending(runs_dir, stem)

    detail = data.run_detail(path_file)
    rollouts = detail.get("rollouts") or []
    if not rollouts:
        # JSONL exists but empty / no rollouts yet — still show progress.
        pending = _page_run_pending(runs_dir, stem)
        # Prefer pending UI with a note that the file is warming up.
        return pending

    summary = detail.get("summary") or {}
    mean = summary.get("mean_reward")
    mean_s = f"{mean:.3f}" if isinstance(mean, float) else "-"
    statuses = summary.get("statuses") or {}
    status_s = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    rows = []
    for r in rollouts:
        status = str(r.get("status") or "?")
        cls = "ok" if status == "completed" else ("bad" if status in {"failed", "aborted"} else "")
        reward = r.get("reward")
        reward_s = "-" if reward is None else str(reward)
        err = r.get("error_code")
        err_s = f" [{err}]" if err else ""
        tools = r.get("tools") or {}
        tool_s = " ".join(f"{k}×{v}" for k, v in tools.items()) or "-"
        rid = r["rollout_id"]
        rows.append(
            "<tr>"
            f"<td><a href='/run/{_esc(stem)}/rollout/{_esc(rid)}'>{_esc(rid)}</a></td>"
            f"<td class='{cls}'>{_esc(status)}{_esc(err_s)}</td>"
            f"<td>{_esc(reward_s)}</td>"
            f"<td>{_esc(_fmt_secs(r.get('wall_seconds')))}</td>"
            f"<td>{_esc(tool_s)}</td>"
            "</tr>"
        )
    body = f"""
    <nav class="crumb"><a href="/">runs</a> / {_esc(detail['name'])}</nav>
    <div class="meta">
      <span>status <strong>{_esc(status_s)}</strong></span>
      <span>mean reward <strong>{_esc(mean_s)}</strong></span>
      <span>n <strong>{len(rollouts)}</strong></span>
    </div>
    <table>
      <thead><tr>
        <th>rollout</th><th>status</th><th>reward</th><th>wall</th><th>tools</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """
    return _layout(detail["name"], body)


def _page_rollout(runs_dir: Path, stem: str, rollout_id: str) -> str:
    detail = data.rollout_detail(_resolve_run(runs_dir, stem), rollout_id)
    decomp = detail.get("wall_decomposition") or {}
    total = sum(float(v) for v in decomp.values()) or 1.0
    bar_bits = []
    mapping = (
        ("inference", "b-inf"),
        ("sandbox", "b-sbx"),
        ("environment", "b-env"),
        ("other", "b-oth"),
    )
    for key, cls in mapping:
        v = float(decomp.get(key) or 0.0)
        if v <= 0:
            continue
        pct = 100.0 * v / total
        bar_bits.append(f'<span class="{cls}" style="width:{pct:.1f}%" title="{key}"></span>')
    legend = " · ".join(
        f"{k} {_fmt_secs(decomp.get(k))} ({100 * float(decomp.get(k) or 0) / total:.0f}%)"
        for k, _ in mapping
        if float(decomp.get(k) or 0) > 0
    )
    steps = []
    for step in detail.get("steps") or []:
        err = step.get("error")
        err_s = f" <span class='bad'>! {_esc(err)}</span>" if err else ""
        steps.append(
            f"<tr><td>{_esc(_fmt_secs(step.get('offset_seconds')))}</td>"
            f"<td>{_esc(step.get('short') or step.get('name'))}</td>"
            f"<td>{_esc(_fmt_secs(step.get('duration_seconds')))}{err_s}</td></tr>"
        )
        preview = step.get("generation_preview")
        if preview:
            steps.append(
                f"<tr><td></td><td colspan='2'><pre class='preview'>{_esc(str(preview)[:240])}</pre></td></tr>"
            )
    status = str(detail.get("status") or "?")
    cls = "ok" if status == "completed" else ""
    body = f"""
    <nav class="crumb">
      <a href="/">runs</a> /
      <a href="/run/{_esc(stem)}">{_esc(detail.get('name'))}</a> /
      {_esc(rollout_id)}
    </nav>
    <div class="meta">
      <span>status <strong class="{cls}">{_esc(status)}</strong></span>
      <span>reward <strong>{_esc(detail.get('reward'))}</strong></span>
      <span>wall <strong>{_esc(_fmt_secs(detail.get('wall_seconds')))}</strong></span>
    </div>
    <div class="bar">{''.join(bar_bits)}</div>
    <p class="meta">{_esc(legend)}</p>
    <table>
      <thead><tr><th>offset</th><th>span</th><th>duration</th></tr></thead>
      <tbody>{''.join(steps) or '<tr><td colspan="3" class="empty">empty timeline</td></tr>'}</tbody>
    </table>
    """
    return _layout(f"{stem} / {rollout_id}", body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Daytona Gym dashboard (local runs/, or pull from a GPU host)"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory of telemetry JSONL files (default: runs)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address (default: 127.0.0.1, or 0.0.0.0 on RunPod)",
    )
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser tab",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        metavar="FILE.html",
        help="Offline HTML dump (last resort; prefer live --share tunnel)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Publish a public Cloudflare quick-tunnel URL",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Do not tunnel (loopback only)",
    )
    parser.add_argument(
        "--remote",
        default=None,
        metavar="USER@HOST",
        help="Escape hatch: pull from explicit SSH host (default: auto RunPod from .env)",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=None,
        help="SSH port for --remote (optional; or user@host:PORT)",
    )
    parser.add_argument(
        "--remote-root",
        default=None,
        help="Remote repo dir (default: /root/daytona-training-gym-spec "
        "or DAYTONA_GYM_REMOTE_REPO)",
    )
    parser.add_argument(
        "--identity",
        "-i",
        type=Path,
        default=None,
        help="SSH private key (or DAYTONA_GYM_SSH_IDENTITY)",
    )
    parser.add_argument(
        "--pull-only",
        action="store_true",
        help="Sync remote runs/ then exit (no HTTP server)",
    )
    parser.add_argument(
        "--remote-interval",
        type=float,
        default=8.0,
        help="Seconds between remote pull loops while serving (default: 8)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Do not auto-pull from RunPod; serve local runs/ only",
    )
    args = parser.parse_args(argv)

    if args.export is not None:
        path = export_static(args.runs_dir, args.export)
        print(f"wrote {path.resolve()}")
        return 0

    remote = resolve_remote_target(args.remote)
    remote_identity = args.identity
    remote_root = args.remote_root
    # On the pod itself runs/ is already local — auto-pull would SSH into ourselves.
    on_gpu_box = detect_serve_context().kind == "runpod"
    if remote is None and not args.local_only and not on_gpu_box:
        auto = resolve_auto_remote()
        if auto is not None:
            remote, auto_ident, auto_root = auto
            if remote_identity is None:
                remote_identity = auto_ident
            if remote_root is None:
                remote_root = auto_root
            print(f"auto remote: {remote}", flush=True)

    stop_sync = threading.Event()

    def _sync_once(*, label: str = "syncing") -> int:
        assert remote is not None
        print(f"{label} runs/ from {remote} …")
        sync_runs_from_ssh(
            target=remote,
            local_runs=args.runs_dir,
            remote_root=remote_root,
            identity=remote_identity,
            ssh_port=args.ssh_port,
        )
        n = len(list(Path(args.runs_dir).glob("*.jsonl")))
        print(f"synced {n} jsonl file(s) → {Path(args.runs_dir).resolve()}")
        return n

    if remote:
        try:
            _sync_once()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.pull_only:
            return 0

        def _sync_loop() -> None:
            while not stop_sync.wait(max(3.0, float(args.remote_interval))):
                try:
                    sync_runs_from_ssh(
                        target=remote,
                        local_runs=args.runs_dir,
                        remote_root=remote_root,
                        identity=remote_identity,
                        ssh_port=args.ssh_port,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"remote sync warning: {exc}", file=sys.stderr)

        threading.Thread(target=_sync_loop, name="dash-remote-sync", daemon=True).start()
        print(
            f"remote pull loop every {args.remote_interval:g}s "
            "(Ctrl+C stops dash + sync)",
            flush=True,
        )

    context = detect_serve_context()
    host = args.host
    if host is None:
        host = "0.0.0.0" if context.kind == "runpod" else "127.0.0.1"

    share: bool | None
    if args.no_share:
        share = False
    elif args.share:
        share = True
    else:
        # Loopback is unreachable from the user's browser on a GPU box, so
        # tunnel there by default (same rule as TrainingRun.open()).
        share = context.kind in {"runpod", "ssh"}

    if spa_available():
        print(f"SPA dashboard  {_STATIC_DIR}", flush=True)
    else:
        print(
            "SPA not built — serving legacy HTML "
            "(cd daytona_gym/telemetry/dashboard_frontend && npm run build)",
            flush=True,
        )

    try:
        serve(
            runs_dir=args.runs_dir,
            host=host,
            port=args.port,
            open_browser=not args.no_open,
            share=share,
        )
    finally:
        stop_sync.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
