"""Minimal local dashboard over ``runs/*.jsonl``.

  dg dash
  dg open
  dg dash --remote user@host          # laptop: pull runs then serve
  DAYTONA_GYM_SSH=user@host dg dash   # same, via env
  python -m daytona_gym.telemetry.dashboard --runs-dir runs --port 8765
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from daytona_gym.telemetry import dashboard_data as data
from daytona_gym.telemetry.dashboard_sync import resolve_remote_target, sync_runs_from_ssh


def serve(
    *,
    runs_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    context = detect_serve_context()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
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

    httpd = ThreadingHTTPServer((host, port), Handler)
    local_url = f"http://127.0.0.1:{port}/"
    print(f"Daytona Gym dashboard  {local_url}")
    print(f"runs dir: {runs_dir}")
    _print_access_hints(context, host=host, port=port)
    print("Ctrl+C to stop")
    should_open = open_browser and context.kind == "local"
    if should_open:
        try:
            webbrowser.open(local_url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        httpd.server_close()


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
    if context.kind == "runpod" and context.runpod_pod_id:
        proxy = f"https://{context.runpod_pod_id}-{port}.proxy.runpod.net"
        print()
        print("Open on your laptop:")
        print(f"  {proxy}")
        print(f"  (In RunPod UI → Connect: expose HTTP port {port} once if that link 502s.)")
        print()
        print("Or SSH tunnel:")
        print(f"  ssh -L {port}:127.0.0.1:{port} <your-runpod-ssh-target>")
        print()
        return
    if context.kind == "ssh":
        print()
        print("Open on your laptop (SSH tunnel):")
        print(f"  ssh -L {port}:127.0.0.1:{port} <same-user-host-you-used>")
        print(f"  then open http://127.0.0.1:{port}/")
        print()
        return
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"listening on {host}:{port}")


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


def _layout(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
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
    if not runs:
        body = (
            f'<p class="empty">No <code>*.jsonl</code> files in '
            f"<code>{_esc(runs_dir)}</code>. Run a dogfood job first.</p>"
        )
        return _layout("Runs", body)
    rows = []
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
    return _layout("Runs", body)


def _page_run(runs_dir: Path, stem: str) -> str:
    detail = data.run_detail(_resolve_run(runs_dir, stem))
    summary = detail.get("summary") or {}
    mean = summary.get("mean_reward")
    mean_s = f"{mean:.3f}" if isinstance(mean, float) else "-"
    statuses = summary.get("statuses") or {}
    status_s = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    rows = []
    for r in detail.get("rollouts") or []:
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
      <span>n <strong>{len(detail.get('rollouts') or [])}</strong></span>
    </div>
    <table>
      <thead><tr>
        <th>rollout</th><th>status</th><th>reward</th><th>wall</th><th>tools</th>
      </tr></thead>
      <tbody>{''.join(rows) or '<tr><td colspan="5" class="empty">no rollouts</td></tr>'}</tbody>
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
        "--remote",
        default=None,
        metavar="USER@HOST",
        help="Pull runs/ from this SSH host first (or set DAYTONA_GYM_SSH)",
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

    remote = resolve_remote_target(args.remote)
    if remote:
        print(f"syncing runs/ from {remote} …")
        try:
            sync_runs_from_ssh(
                target=remote,
                local_runs=args.runs_dir,
                remote_root=args.remote_root,
                identity=args.identity,
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

    serve(
        runs_dir=args.runs_dir,
        host=host,
        port=args.port,
        open_browser=not args.no_open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
