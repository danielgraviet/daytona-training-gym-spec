"""CPU tests for RunPod → SshWorker resolution (no live API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daytona_gym.gym.providers.runpod import parse_runpod_ssh, resolve_runpod_ssh, runpod_worker
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def test_parse_legacy_port_mappings() -> None:
    info = parse_runpod_ssh(
        {
            "id": "abc123",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 12713},
        }
    )
    assert info is not None
    assert info.public_ip == "1.2.3.4"
    assert info.ssh_port == 12713
    w = info.to_worker(identity="/tmp/key")
    assert w.host == "root@1.2.3.4"
    assert w.port == 12713


def test_parse_runtime_ports() -> None:
    info = parse_runpod_ssh(
        {
            "id": "pod9",
            "runtime": {
                "ports": [
                    {"private": 8888, "public": 1, "ip": "9.9.9.9", "type": "http"},
                    {"private": 22, "public": 2222, "ip": "8.8.8.8", "type": "tcp"},
                ]
            },
        }
    )
    assert info is not None
    assert info.public_ip == "8.8.8.8"
    assert info.ssh_port == 2222


def test_parse_missing_returns_none() -> None:
    assert parse_runpod_ssh({"id": "x"}) is None


def test_resolve_uses_http(monkeypatch) -> None:
    payload = {"id": "pod1", "publicIp": "10.0.0.1", "portMappings": {"22": 9999}}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        assert "pods/pod1" in req.full_url
        assert req.headers.get("Authorization") == "Bearer secret"
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    info = resolve_runpod_ssh("pod1")
    assert info.ssh_port == 9999
    w = runpod_worker("pod1", identity="/tmp/k")
    assert w.port == 9999


def test_resolve_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(DaytonaError) as caught:
        resolve_runpod_ssh("pod1")
    assert caught.value.code == ErrorCode.USER_CODE_ERROR
