"""CPU tests for RunPod → SshWorker resolution (no live API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daytona_gym.gym.providers.runpod import parse_runpod_ssh, resolve_runpod_ssh, runpod_worker
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def test_parse_v2_proxy_and_direct() -> None:
    info = parse_runpod_ssh(
        {
            "id": "abc123",
            "ssh": {
                "proxy": {
                    "host": "ssh.runpod.io",
                    "port": 22,
                    "username": "abc123-deadbeef",
                },
                "direct": {
                    "host": "1.2.3.4",
                    "port": 12713,
                    "username": "root",
                },
            },
        }
    )
    assert info.proxy_user == "abc123-deadbeef"
    assert info.public_ip == "1.2.3.4"
    assert info.ssh_port == 12713
    w = info.to_worker(identity="/tmp/key", prefer="proxy")
    assert w.host == "abc123-deadbeef@ssh.runpod.io"
    assert w.transport == "shell"
    w2 = info.to_worker(identity="/tmp/key", prefer="direct")
    assert w2.host == "root@1.2.3.4"
    assert w2.port == 12713


def test_parse_legacy_port_mappings() -> None:
    info = parse_runpod_ssh(
        {
            "id": "abc123",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 12713},
        }
    )
    assert info.public_ip == "1.2.3.4"
    assert info.ssh_port == 12713
    assert info.proxy_user is None


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
    assert info.public_ip == "8.8.8.8"
    assert info.ssh_port == 2222


def test_resolve_uses_v2_http(monkeypatch) -> None:
    payload = {
        "id": "pod1",
        "ssh": {
            "proxy": {"username": "pod1-token", "host": "ssh.runpod.io", "port": 22},
            "direct": {"host": "10.0.0.1", "port": 9999, "username": "root"},
        },
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        assert "api.runpod.io/v2/pods/pod1" in req.full_url
        headers = {k.lower(): v for k, v in req.header_items()}
        assert headers.get("authorization") == "Bearer secret"
        assert headers.get("user-agent", "").startswith("daytona-gym/")
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    info = resolve_runpod_ssh("pod1")
    assert info.proxy_user == "pod1-token"
    w = runpod_worker("pod1", identity="/tmp/k")
    assert w.host == "pod1-token@ssh.runpod.io"
    assert w.transport == "shell"


def test_resolve_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(DaytonaError) as caught:
        resolve_runpod_ssh("pod1")
    assert caught.value.code == ErrorCode.USER_CODE_ERROR
