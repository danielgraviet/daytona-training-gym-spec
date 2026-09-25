"""Tests for Modal-shaped dataset materialization + Harbor seed rows."""

from __future__ import annotations

from pathlib import Path

from daytona_gym.adapters.slime._coding_seed import seed_files_from_label
from daytona_gym.adapters.slime.generate import _resolve_seed_files
from daytona_gym.gym.dataset import (
    HarborDataset,
    HuggingFaceDataset,
    PromptJsonlDataset,
    deserialize_dataset,
    materialize_dataset,
    serialize_dataset,
)


def test_prompt_jsonl_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "a.jsonl"
    src.write_text('{"prompt":"hi","label":"x"}\n', encoding="utf-8")
    ds = PromptJsonlDataset(src)
    blob = serialize_dataset(ds)
    assert blob["kind"] == "prompt_jsonl"
    again = deserialize_dataset(blob)
    assert isinstance(again, PromptJsonlDataset)
    rows = list(again.rows())
    assert rows[0]["prompt"] == "hi"


def test_harbor_local_pack_writes_seed_files(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "examples/gym_sdk/harbor_tasks"
    assert root.is_dir(), f"missing fixture pack: {root}"
    ds = HarborDataset(path=root, train_size=2, shuffle_seed=0, shuffle_tasks=False)
    out = materialize_dataset(ds, runs_dir=tmp_path)
    assert out.resolved_path().is_file()
    rows = list(out.rows())
    assert len(rows) >= 1
    label = rows[0]["label"]
    assert isinstance(label, dict)
    assert label.get("seed_files")
    assert any(k.endswith("solution.py") or k == "solution.py" for k in label["seed_files"])
    seeds = seed_files_from_label(label)
    assert seeds


def test_generate_prefers_sample_seed_over_profile() -> None:
    class _Sample:
        label = {
            "seed_files": {"solution.py": "def two_sum(...):\n    pass\n"},
            "run_tests_command": "python tests/test_solution.py",
        }

    class _Args:
        daytona_seed_files = {"broken.py": "old"}

    files = _resolve_seed_files(_Args(), _Sample())
    assert "solution.py" in files
    assert "broken.py" not in files


def test_hf_serialize_shape() -> None:
    ds = HuggingFaceDataset(
        hf_repo="openai/gsm8k",
        hf_split="train[:8]",
        input_column="question",
        output_column="answer",
    )
    blob = serialize_dataset(ds)
    assert blob["kind"] == "huggingface"
    assert blob["hf_repo"] == "openai/gsm8k"
    again = deserialize_dataset(blob)
    assert isinstance(again, HuggingFaceDataset)
    assert again.input_column == "question"
