"""CPU tests for Vast.ai → SshWorker resolution (no live API)."""

from __future__ import annotations

import json

import pytest

from daytona_gym.gym.providers.vast import parse_vast_ssh, resolve_vast_ssh, search_offers
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def test_parse_vast_ssh() -> None:
    info = parse_vast_ssh(
        {
            "id": 42,
            "ssh_host": "ssh4.vast.ai",
            "ssh_port": 12345,
            "ssh_user": "root",
            "actual_status": "running",
            "gpu_name": "RTX_4090",
        }
    )
    assert info.instance_id == 42
    assert info.host == "ssh4.vast.ai"
    assert info.port == 12345
    w = info.to_worker(identity="/tmp/key")
    assert w.host == "root@ssh4.vast.ai"
    assert w.port == 12345
    assert w.name == "vast:42"


def test_parse_missing_ssh_raises() -> None:
    with pytest.raises(DaytonaError) as caught:
        parse_vast_ssh({"id": 1, "actual_status": "loading"})
    assert caught.value.code is ErrorCode.PLATFORM_ERROR


def test_search_offers_posts_filters(monkeypatch) -> None:
    captured: dict = {}

    def fake_http(method, path, *, api_key, body=None, query=None):
        captured.update(method=method, path=path, body=body, api_key=api_key)
        return {
            "offers": [
                {"id": 9, "dph_total": 0.4, "gpu_name": "RTX_4090"},
                {"id": 8, "dph_total": 0.3, "gpu_name": "RTX_4090"},
            ]
        }

    monkeypatch.setattr(
        "daytona_gym.gym.providers.vast._http", fake_http
    )
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    offers = search_offers(gpu_ram_gb=24, cpu_ram_gb=32)
    assert captured["method"] == "POST"
    assert captured["path"] == "/bundles/"
    assert captured["body"]["gpu_ram"]["gte"] == 24 * 1024
    assert captured["body"]["cpu_ram"]["gte"] == 32 * 1024
    assert len(offers) == 2


def test_resolve_vast_ssh_uses_get(monkeypatch) -> None:
    def fake_http(method, path, *, api_key, body=None, query=None):
        assert method == "GET"
        assert path == "/instances/99/"
        return {
            "id": 99,
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "actual_status": "running",
        }

    monkeypatch.setattr("daytona_gym.gym.providers.vast._http", fake_http)
    monkeypatch.setenv("VAST_API_KEY", "k")
    info = resolve_vast_ssh(99)
    assert info.port == 2222
    assert info.host == "1.2.3.4"
