"""Minimal local dashboard over ``runs/*.jsonl``.

  dg dash                 # local browser; on GPU: live public URL via tunnel
  dg open
  dg dash --share         # force Cloudflare quick tunnel
  dg dash --no-share      # loopback only
  python -m daytona_gym.telemetry.dashboard --runs-dir runs --port 8765
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from daytona_gym.telemetry import dashboard_data as data
from daytona_gym.telemetry import progress as run_progress
from daytona_gym.telemetry.dashboard_sync import (
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
        share = context.kind in {"runpod", "ssh"}

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
            self.end_headers()
            self.wfile.write(payload)

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
    if context.kind in {"runpod", "ssh"}:
        print()
        print("Tip: on a GPU box, omit --no-share so dg dash prints a live public URL.")
        print()
        return
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"listening on {host}:{port}")


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
    if path in {"/", "/index.html"}:
        return _page_index(runs_dir), "text/html; charset=utf-8"
    if path == "/api/runs" or path.startswith("/api/runs/"):
        return _api(path, runs_dir), "application/json; charset=utf-8"
    if path.startswith("/run/"):
        parts = [p for p in path.split("/") if p]
        # /run/<stem> or /run/<stem>/rollout/<id>
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
    # api / runs
    if parts == ["api", "runs"]:
        return json.dumps(data.list_run_files(runs_dir))
    # api / runs / <stem>
    if len(parts) == 3:
        path_file = _resolve_run(runs_dir, parts[2])
        return json.dumps(data.run_detail(path_file))
    # api / runs / <stem> / rollouts / <id>
    if len(parts) == 5 and parts[3] == "rollouts":
        path_file = _resolve_run(runs_dir, parts[2])
        return json.dumps(data.rollout_detail(path_file, parts[4]))
    raise FileNotFoundError(path)


def _resolve_run(runs_dir: Path, stem: str) -> Path:
    candidate = runs_dir / f"{stem}.jsonl"
    if candidate.is_file():
        return candidate
    # allow full filename
    alt = runs_dir / stem
    if alt.is_file():
        return alt
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


def _layout(title: str, body: str, *, refresh_seconds: int | None = None) -> str:
    refresh = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}"/>'
        if refresh_seconds
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  {refresh}
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
    .phases .tag {{
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 0.75rem;
      color: var(--accent);
      margin-right: 0.5rem;
    }}
    pre.preview {{
      margin: 0.2rem 0 0.6rem 1rem;
      white-space: pre-wrap;
      color: var(--muted);
      font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
      font-size: 0.78rem;
    }}
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
        rows.append(
            "<tr>"
            f"<td><a href='/run/{_esc(stem)}'>{_esc(stem)}</a></td>"
            "<td>0</td>"
            f"<td class='ok'>starting · {_esc(prog.get('phase') or '…')}</td>"
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
    """
    return _layout("Runs", body, refresh_seconds=5 if progress_only else None)


_PHASE_STEPS = (
    ("starting", "Run created"),
    ("model_download", "Download model weights"),
    ("model_convert", "Convert HF → Megatron"),
    ("ray_start", "Start Ray / Slime"),
    ("training", "Training — waiting for first rollout"),
    ("live", "Rollouts live"),
)


def _page_run_pending(runs_dir: Path, stem: str) -> str:
    prog = run_progress.read_progress(runs_dir, stem) or {}
    phase = str(prog.get("phase") or "starting")
    message = str(prog.get("message") or "Worker is starting…")
    items = []
    seen_active = False
    for key, label in _PHASE_STEPS:
        if key == "live":
            cls = ""
        elif key == phase:
            cls = "active"
            seen_active = True
        elif not seen_active:
            cls = "done"
        else:
            cls = ""
        tag = "now" if cls == "active" else ("ok" if cls == "done" else "")
        items.append(
            f'<li class="{cls}">'
            + (f'<span class="tag">{tag}</span>' if tag else "")
            + f"{_esc(label)}</li>"
        )
    body = f"""
    <nav class="crumb"><a href="/">runs</a> / {_esc(stem)}</nav>
    <div class="meta">
      <span>status <strong>starting</strong></span>
      <span>{_esc(message)}</span>
    </div>
    <p>This page refreshes every few seconds. Rollout timelines (prefill, decode,
    sandbox tools) appear once training emits telemetry.</p>
    <ul class="phases">{"".join(items)}</ul>
    """
    return _layout(f"{stem} (starting)", body, refresh_seconds=3)


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
    parser.add_argument("--port", type=int, default=8765)
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
        help="Pull runs/ first (optional)",
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
        help="Remote repo dir under $HOME (default: daytona-training-gym-spec "
        "or DAYTONA_GYM_SSH_ROOT)",
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
    args = parser.parse_args(argv)

    if args.export is not None:
        path = export_static(args.runs_dir, args.export)
        print(f"wrote {path.resolve()}")
        return 0

    remote = resolve_remote_target(args.remote)
    if remote:
        print(f"syncing runs/ from {remote} …")
        try:
            sync_runs_from_ssh(
                target=remote,
                local_runs=args.runs_dir,
                remote_root=args.remote_root,
                identity=args.identity,
                ssh_port=args.ssh_port,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        n = len(list(Path(args.runs_dir).glob("*.jsonl")))
        print(f"synced {n} jsonl file(s) → {Path(args.runs_dir).resolve()}")
        if args.pull_only:
            return 0

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
        share = None  # auto: on for runpod/ssh

    serve(
        runs_dir=args.runs_dir,
        host=host,
        port=args.port,
        open_browser=not args.no_open,
        share=share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
