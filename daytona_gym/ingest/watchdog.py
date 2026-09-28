"""``dg ingest watch`` — reopen a stopped Daytona ingest sandbox.

Preview traffic is not "activity", so the sandbox can still stop despite
``auto_stop_interval=0`` (platform/org limits). This loop polls ``/healthz``
and re-runs ``dg ingest deploy --daytona`` when the host is unreachable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

from daytona_gym.ingest.config import INGEST_TOKEN_ENV, INGEST_URL_ENV, IngestConfig


def health_ok(url: str, *, timeout: float = 10.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=timeout) as resp:  # noqa: S310
            return int(resp.status) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def redeploy(*, name: str, ref: str) -> tuple[str, str]:
    from daytona_gym.ingest.deploy import deploy, resolve_spec
    import asyncio

    token = (os.environ.get(INGEST_TOKEN_ENV) or "").strip() or None
    spec = resolve_spec(ref)
    _sid, url, new_token = asyncio.run(deploy(name=name, spec=spec, token=token))
    os.environ[INGEST_URL_ENV] = url
    os.environ[INGEST_TOKEN_ENV] = new_token
    return url, new_token


def watch_loop(
    *,
    interval: float,
    name: str,
    ref: str,
    once: bool = False,
) -> int:
    cfg = IngestConfig.from_env()
    url = cfg.url if cfg is not None else None
    failures = 0
    while True:
        target = url or (os.environ.get(INGEST_URL_ENV) or "").strip()
        if not target:
            print("[watch] no ingest URL — deploying…", flush=True)
            try:
                url, _token = redeploy(name=name, ref=ref)
                print(f"[watch] deployed {url}", flush=True)
                failures = 0
            except Exception as exc:  # noqa: BLE001
                print(f"[watch] deploy failed: {exc}", file=sys.stderr)
                failures += 1
        elif health_ok(target):
            if failures:
                print(f"[watch] healthy again: {target}", flush=True)
            failures = 0
        else:
            failures += 1
            print(
                f"[watch] {target}/healthz failed (n={failures}) — redeploying…",
                flush=True,
            )
            try:
                url, _token = redeploy(name=name, ref=ref)
                print(f"[watch] redeployed {url}", flush=True)
                failures = 0
            except Exception as exc:  # noqa: BLE001
                print(f"[watch] redeploy failed: {exc}", file=sys.stderr)
        if once:
            return 0 if failures == 0 else 1
        time.sleep(max(5.0, float(interval)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dg ingest watch",
        description="Poll ingest /healthz and redeploy the Daytona sandbox when down.",
    )
    parser.add_argument(
        "--daytona",
        action="store_true",
        required=True,
        help="Watch/redeploy the Daytona-hosted ingest sandbox",
    )
    parser.add_argument("--name", default="daytona-gym-ingest")
    parser.add_argument("--ref", default="main")
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between health checks (default 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single check + optional redeploy, then exit",
    )
    args = parser.parse_args(argv)
    if not os.environ.get("DAYTONA_API_KEY"):
        print("DAYTONA_API_KEY is required", file=sys.stderr)
        return 2
    try:
        return watch_loop(
            interval=args.interval,
            name=args.name,
            ref=args.ref,
            once=bool(args.once),
        )
    except KeyboardInterrupt:
        print("\n[watch] stopped", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
