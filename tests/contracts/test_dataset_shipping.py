"""3.3: local datasets travel with the launch instead of assuming the worker's git clone."""

from __future__ import annotations

import base64
import gzip
import json
import re
from pathlib import Path

import pytest

from daytona_gym.gym import dataset as ds
from daytona_gym.gym.dataset import (
    HarborDataset,
    HuggingFaceDataset,
    PromptJsonlDataset,
    inline_dataset_blob,
    materialize_inline,
)
from daytona_gym.runtime.errors import DaytonaError

REPO = Path(__file__).resolve().parents[2]


def _jsonl(tmp_path: Path, rows: int = 3) -> Path:
    path = tmp_path / "my tasks.jsonl"  # space: names get sanitized on the worker
    path.write_text("".join(json.dumps({"prompt": f"p{i}", "label": str(i)}) + "\n" for i in range(rows)))
    return path


def test_local_jsonl_roundtrips_through_the_worker(tmp_path) -> None:
    src = _jsonl(tmp_path)
    blob = inline_dataset_blob(PromptJsonlDataset(src))
    assert blob["kind"] == "inline_jsonl" and blob["name"] == "my tasks.jsonl"

    worker_runs = tmp_path / "worker" / "runs"
    out = materialize_inline(json.loads(json.dumps(blob)), runs_dir=worker_runs)
    written = out.resolved_path()
    assert written.parent == worker_runs / "data"
    assert re.fullmatch(r"inline_[0-9a-f]{12}_my_tasks\.jsonl", written.name)
    assert written.read_bytes() == src.read_bytes()
    # Idempotent: materializing again reuses the verified file.
    assert materialize_inline(blob, runs_dir=worker_runs).resolved_path() == written


def test_tampered_payload_is_rejected(tmp_path) -> None:
    blob = inline_dataset_blob(PromptJsonlDataset(_jsonl(tmp_path)))
    blob["content"] += '{"prompt":"injected"}\n'
    with pytest.raises(DaytonaError, match="checksum"):
        materialize_inline(blob, runs_dir=tmp_path / "runs")


def test_oversized_dataset_fails_with_guidance(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ds, "INLINE_MAX_BYTES", 10)
    with pytest.raises(DaytonaError, match="HuggingFaceDataset"):
        inline_dataset_blob(PromptJsonlDataset(_jsonl(tmp_path)))


def test_local_harbor_tasks_are_materialized_and_shipped(tmp_path) -> None:
    harbor = HarborDataset(path=REPO / "examples" / "gym_sdk" / "harbor_tasks")
    blob = inline_dataset_blob(harbor)
    assert blob is not None and blob["kind"] == "inline_jsonl"
    rows = [json.loads(line) for line in blob["content"].splitlines()]
    assert len(rows) == 2  # two_sum + parens
    assert all("seed_files" in json.dumps(r) for r in rows)


def test_remote_sources_are_still_fetched_on_the_worker(tmp_path) -> None:
    assert inline_dataset_blob(HuggingFaceDataset(hf_repo="openai/gsm8k", input_column="question")) is None
    assert inline_dataset_blob(HarborDataset(dataset_name="terminal-bench")) is None
    # A path that only exists on the worker is sent as a path, unchanged.
    assert inline_dataset_blob(PromptJsonlDataset("/root/only-on-worker.jsonl")) is None


def test_pty_upload_is_compressed_owner_only_and_decodes(tmp_path) -> None:
    from daytona_gym.gym.worker import _upload_payload_via_shell

    class Shell:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def run(self, command: str, **_: object) -> str:
            self.commands.append(command)
            return ""

    big = {"dataset": {"content": "".join(json.dumps({"prompt": "same text " * 20}) + "\n" for _ in range(2000))}}
    shell = Shell()
    _upload_payload_via_shell(shell, "/tmp/job.json", big)

    chunks = [c.split("'")[3] for c in shell.commands if c.startswith("printf")]
    raw_size = len(json.dumps(big, separators=(",", ":")))
    assert sum(map(len, chunks)) < raw_size / 5  # repetitive JSONL compresses well
    assert json.loads(gzip.decompress(base64.b64decode("".join(chunks)))) == big
    assert shell.commands[0].startswith("(umask 077") and "umask 077" in shell.commands[-1]
