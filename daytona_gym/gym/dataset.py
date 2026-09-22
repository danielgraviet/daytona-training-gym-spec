from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptJsonlDataset:
    """JSONL prompts with Slime ``--input-key`` / ``--label-key`` (coding dogfood shape)."""

    path: str | Path
    input_key: str = "prompt"
    label_key: str = "label"

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()
