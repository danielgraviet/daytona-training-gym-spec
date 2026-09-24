"""CPU tests for model prep helpers."""

from __future__ import annotations

from pathlib import Path

from daytona_gym.gym.model_prep import _looks_like_hf_checkpoint, _looks_like_torch_dist


def test_looks_like_hf_checkpoint(tmp_path: Path) -> None:
    d = tmp_path / "hf"
    d.mkdir()
    assert not _looks_like_hf_checkpoint(d)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert _looks_like_hf_checkpoint(d)


def test_looks_like_torch_dist(tmp_path: Path) -> None:
    d = tmp_path / "dist"
    d.mkdir()
    assert not _looks_like_torch_dist(d)
    (d / "mp_rank_00").mkdir()
    assert _looks_like_torch_dist(d)
