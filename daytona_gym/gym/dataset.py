"""Modal-shaped dataset configs — materialize on the GPU worker, not the laptop."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DatasetRow = dict[str, Any]


def _fingerprint(fields: dict[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@runtime_checkable
class DatasetConfig(Protocol):
    """Shared dataset materialization (HF / Harbor / local JSONL)."""

    def cache_key(self) -> str | None: ...

    def rows(self) -> Iterable[DatasetRow]: ...

    def write(self, path: str | Path) -> Path: ...

    def serialize(self) -> dict[str, Any]: ...

    @property
    def input_key(self) -> str: ...

    @property
    def label_key(self) -> str: ...


def _write_rows(rows: Iterable[DatasetRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@dataclass(frozen=True)
class PromptJsonlDataset:
    """Local/path escape hatch: JSONL already on disk (or in the cloned repo)."""

    path: str | Path
    input_key: str = "prompt"
    label_key: str = "label"

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()

    def cache_key(self) -> str | None:
        return None

    def rows(self) -> Iterable[DatasetRow]:
        with self.resolved_path().open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row

    def write(self, path: str | Path) -> Path:
        dest = Path(path).expanduser()
        src = self.resolved_path()
        if dest.resolve() == src:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "prompt_jsonl",
            "path": str(self.path),
            "input_key": self.input_key,
            "label_key": self.label_key,
        }

    def to_prompt_jsonl(self, *, out_path: str | Path | None = None) -> PromptJsonlDataset:
        if out_path is None:
            return self
        return PromptJsonlDataset(
            path=self.write(out_path),
            input_key=self.input_key,
            label_key=self.label_key,
        )


@dataclass
class HuggingFaceDataset:
    """Load a Hub split on the **worker** and write JSONL for Slime."""

    hf_repo: str = ""
    hf_split: str = "train"
    hf_config: str | None = None
    input_column: str = "prompt"
    output_column: str | None = "label"
    input_key: str = "prompt"
    label_key: str = "label"
    limit: int | None = None
    system_prompt: str = ""
    prompt_template: str = "{input}"
    out_path: str | Path = "/tmp/daytona_gym_hf_dataset.jsonl"
    # Back-compat aliases from earlier stubs / plan wording.
    repo_id: str | None = None
    split: str | None = None
    prompt_column: str | None = None
    label_column: str | None = None

    def __post_init__(self) -> None:
        if not self.hf_repo and self.repo_id:
            self.hf_repo = self.repo_id
        if self.repo_id is None:
            self.repo_id = self.hf_repo
        if self.split:
            self.hf_split = self.split
        if self.prompt_column:
            self.input_column = self.prompt_column
        if self.label_column is not None:
            self.output_column = self.label_column
        if not self.hf_repo:
            raise ValueError("HuggingFaceDataset requires hf_repo=... (or repo_id=...)")

    def cache_key(self) -> str | None:
        return "hf_" + _fingerprint(
            {
                "hf_repo": self.hf_repo,
                "hf_split": self.hf_split,
                "hf_config": self.hf_config,
                "input_column": self.input_column,
                "output_column": self.output_column,
                "limit": self.limit,
                "prompt_template": self.prompt_template,
            }
        )

    def _load(self):
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceDataset requires the `datasets` package "
                "(pip install datasets)"
            ) from exc
        return load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)

    def rows(self) -> Iterable[DatasetRow]:
        ds = self._load()
        n = 0
        for row in ds:
            if self.limit is not None and n >= int(self.limit):
                break
            if not isinstance(row, dict):
                row = dict(row)
            raw = row.get(self.input_column)
            if raw is None:
                continue
            text = self.prompt_template.format(input=raw)
            if self.system_prompt:
                text = f"{self.system_prompt}\n\n{text}"
            payload: DatasetRow = {self.input_key: text}
            if self.output_column and self.output_column in row:
                payload[self.label_key] = row[self.output_column]
            yield payload
            n += 1

    def write(self, path: str | Path) -> Path:
        return _write_rows(self.rows(), Path(path).expanduser())

    def materialize(self, *, out_path: str | Path | None = None) -> PromptJsonlDataset:
        dest = Path(out_path or self.out_path)
        return self.to_prompt_jsonl(out_path=dest)

    def to_prompt_jsonl(self, *, out_path: str | Path) -> PromptJsonlDataset:
        written = self.write(out_path)
        return PromptJsonlDataset(
            path=written,
            input_key=self.input_key,
            label_key=self.label_key,
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "huggingface",
            "hf_repo": self.hf_repo,
            "hf_split": self.hf_split,
            "hf_config": self.hf_config,
            "input_column": self.input_column,
            "output_column": self.output_column,
            "input_key": self.input_key,
            "label_key": self.label_key,
            "limit": self.limit,
            "system_prompt": self.system_prompt,
            "prompt_template": self.prompt_template,
        }


@dataclass
class HarborDataset:
    """Pull Harbor tasks; each row = instruction + seed/test files for coding.

    Uses ``harbor datasets download`` / ``uvx harbor`` when ``dataset_name`` is
    set. Local ``path`` / ``task_root`` work offline for fixtures.
    """

    dataset_name: str = ""
    path: str | Path | None = None
    task_root: str | Path | None = None
    task_glob: str = "*"
    task_names: list[str] | None = None
    instruction_path: str = "instruction.md"
    test_data_dir: str | None = "tests"
    seed_glob: str = "*.py"
    run_tests_command: str | None = None
    train_size: int | None = None
    shuffle_seed: int = 0
    shuffle_tasks: bool = True
    input_key: str = "prompt"
    label_key: str = "label"
    prompt_template: str = "{instruction}"

    def cache_key(self) -> str | None:
        return "harbor_" + _fingerprint(
            {
                "dataset_name": self.dataset_name,
                "path": str(self.path or ""),
                "task_root": str(self.task_root or ""),
                "task_glob": self.task_glob,
                "task_names": self.task_names,
                "train_size": self.train_size,
                "shuffle_seed": self.shuffle_seed,
            }
        )

    def _harbor_ref(self) -> str:
        if "@" in self.dataset_name:
            return self.dataset_name
        return f"{self.dataset_name}@latest"

    def _cache_dir(self) -> Path:
        slug = self._harbor_ref().replace("/", "--").replace("@", "--")
        return Path.home() / ".cache" / "daytona_gym" / "harbor" / slug

    def _download(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ref = self._harbor_ref()
        harbor_bin = shutil.which("harbor")
        if harbor_bin is not None:
            cmd = [
                harbor_bin,
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        else:
            uvx = shutil.which("uvx")
            if uvx is None:
                raise FileNotFoundError(
                    "Harbor CLI not found. Install `harbor` or `uvx`, or pass "
                    "path=/task_root for a local Harbor task pack."
                )
            cmd = [
                uvx,
                "harbor",
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        import subprocess

        subprocess.run(cmd, check=True)

    def _discover_task_root(self, search_root: Path) -> Path:
        dirs = sorted(
            {p.parent for p in search_root.rglob(self.instruction_path) if p.is_file()}
        )
        if not dirs:
            return search_root
        if len(dirs) == 1:
            return dirs[0].parent
        return Path(os.path.commonpath([str(p) for p in dirs]))

    def _resolve_task_root(self) -> Path:
        if self.path:
            root = Path(self.path).expanduser().resolve()
        elif self.task_root:
            root = Path(self.task_root).expanduser().resolve()
        elif self.dataset_name:
            cache = self._cache_dir()
            if not any(cache.rglob(self.instruction_path)):
                self._download(cache)
            root = self._discover_task_root(cache)
        else:
            raise ValueError("HarborDataset requires dataset_name, path, or task_root")
        if not root.is_dir():
            raise FileNotFoundError(f"Harbor task root missing: {root}")
        return root

    def _task_dirs(self) -> list[Path]:
        root = self._resolve_task_root()
        if self.task_names is not None:
            dirs = [(root / n).resolve() for n in self.task_names if (root / n).is_dir()]
        else:
            dirs = sorted(p.resolve() for p in root.glob(self.task_glob) if p.is_dir())
        if not dirs:
            discovered = self._discover_task_root(root)
            if discovered != root:
                dirs = sorted(
                    p.resolve() for p in discovered.glob(self.task_glob) if p.is_dir()
                )
        if self.shuffle_tasks:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(dirs)
        if self.train_size is not None:
            dirs = dirs[: int(self.train_size)]
        if not dirs:
            raise FileNotFoundError(f"No Harbor tasks under {root}")
        return dirs

    def _collect_seed_files(self, task_dir: Path) -> dict[str, str]:
        files: dict[str, str] = {}
        candidates: list[Path] = []
        for sub in ("", "environment", "src", "solution", "workspace"):
            base = task_dir if not sub else task_dir / sub
            if base.is_dir():
                candidates.extend(sorted(base.glob(self.seed_glob)))
        if self.test_data_dir:
            tests = task_dir / self.test_data_dir
            if tests.is_dir():
                for path in sorted(tests.rglob("*")):
                    if path.is_file() and path.suffix in {
                        ".py",
                        ".txt",
                        ".sh",
                        ".in",
                        ".out",
                    }:
                        candidates.append(path)
        env_dir = task_dir / "environment"
        if env_dir.is_dir():
            for path in sorted(env_dir.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".txt", ".sh", ".md"}:
                    candidates.append(path)
        for path in candidates:
            try:
                rel = path.relative_to(task_dir).as_posix()
            except ValueError:
                rel = path.name
            try:
                files[rel] = path.read_text(encoding="utf-8")
            except OSError:
                continue
        return files

    def _default_run_tests(self, seed_files: dict[str, str]) -> str:
        if self.run_tests_command:
            return self.run_tests_command
        for name in (
            "tests/test.sh",
            "test.sh",
            "tests/run_tests.py",
            "tests/test_solution.py",
            "test_broken.py",
            "tests/test_broken.py",
        ):
            if name in seed_files:
                if name.endswith(".sh"):
                    return f"bash {name}"
                return f"python {name}"
        for key in seed_files:
            base = Path(key).name
            if base.startswith("test_") and base.endswith(".py"):
                return f"python {key}"
            if base == "test.sh":
                return f"bash {key}"
        return "python test_broken.py"

    def rows(self) -> Iterable[DatasetRow]:
        for task_dir in self._task_dirs():
            instruction_file = task_dir / self.instruction_path
            if not instruction_file.is_file():
                alt = task_dir / "instruction.txt"
                if alt.is_file():
                    instruction_file = alt
                else:
                    continue
            instruction = instruction_file.read_text(encoding="utf-8")
            prompt = self.prompt_template.format(instruction=instruction)
            seed_files = self._collect_seed_files(task_dir)
            run_cmd = self._default_run_tests(seed_files)
            label: dict[str, Any] = {
                "harbor_task_name": task_dir.name,
                "harbor_task_path": task_dir.as_posix(),
                "seed_files": seed_files,
                "run_tests_command": run_cmd,
            }
            yield {self.input_key: prompt, self.label_key: label}

    def write(self, path: str | Path) -> Path:
        return _write_rows(self.rows(), Path(path).expanduser())

    def to_prompt_jsonl(self, *, out_path: str | Path) -> PromptJsonlDataset:
        written = self.write(out_path)
        return PromptJsonlDataset(
            path=written,
            input_key=self.input_key,
            label_key=self.label_key,
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "kind": "harbor",
            "dataset_name": self.dataset_name,
            "path": str(self.path) if self.path else None,
            "task_root": str(self.task_root) if self.task_root else None,
            "task_glob": self.task_glob,
            "task_names": self.task_names,
            "instruction_path": self.instruction_path,
            "test_data_dir": self.test_data_dir,
            "seed_glob": self.seed_glob,
            "run_tests_command": self.run_tests_command,
            "train_size": self.train_size,
            "shuffle_seed": self.shuffle_seed,
            "input_key": self.input_key,
            "label_key": self.label_key,
        }


AnyDataset = PromptJsonlDataset | HuggingFaceDataset | HarborDataset


def deserialize_dataset(blob: dict[str, Any]) -> AnyDataset:
    kind = str(blob.get("kind") or blob.get("type") or "prompt_jsonl")
    if kind in {"prompt_jsonl", "jsonl", "default"}:
        return PromptJsonlDataset(
            path=blob["path"],
            input_key=blob.get("input_key", "prompt"),
            label_key=blob.get("label_key", "label"),
        )
    if kind in {"huggingface", "hugging_face", "hf"}:
        return HuggingFaceDataset(
            hf_repo=blob.get("hf_repo") or blob.get("repo_id") or "",
            hf_split=blob.get("hf_split", blob.get("split", "train")),
            hf_config=blob.get("hf_config"),
            input_column=blob.get("input_column", blob.get("prompt_column", "prompt")),
            output_column=blob.get("output_column", blob.get("label_column", "label")),
            input_key=blob.get("input_key", "prompt"),
            label_key=blob.get("label_key", "label"),
            limit=blob.get("limit"),
            system_prompt=blob.get("system_prompt", ""),
            prompt_template=blob.get("prompt_template", "{input}"),
        )
    if kind == INLINE_KIND:
        raise ValueError("inline_jsonl datasets must be materialized with materialize_inline()")
    if kind == "harbor":
        return HarborDataset(
            dataset_name=blob.get("dataset_name", ""),
            path=blob.get("path"),
            task_root=blob.get("task_root"),
            task_glob=blob.get("task_glob", "*"),
            task_names=blob.get("task_names"),
            instruction_path=blob.get("instruction_path", "instruction.md"),
            test_data_dir=blob.get("test_data_dir", "tests"),
            seed_glob=blob.get("seed_glob", "*.py"),
            run_tests_command=blob.get("run_tests_command"),
            train_size=blob.get("train_size"),
            shuffle_seed=int(blob.get("shuffle_seed", 0)),
            input_key=blob.get("input_key", "prompt"),
            label_key=blob.get("label_key", "label"),
        )
    raise ValueError(f"unknown dataset kind: {kind!r}")


def serialize_dataset(dataset: AnyDataset) -> dict[str, Any]:
    return dataset.serialize()


def materialize_dataset(
    dataset: AnyDataset | dict[str, Any],
    *,
    runs_dir: Path,
) -> PromptJsonlDataset:
    """Write dataset JSONL under ``runs_dir/data/`` when needed; return PromptJsonl."""
    if isinstance(dataset, dict):
        dataset = deserialize_dataset(dataset)
    if isinstance(dataset, PromptJsonlDataset):
        path = dataset.resolved_path()
        if path.is_file():
            return dataset
        # Remote-shaped paths may not exist until the worker clones the repo;
        # keep the handle and let validate() catch missing files at train time.
        return dataset

    data_dir = Path(runs_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    key = dataset.cache_key() or "dataset"
    out = data_dir / f"{key}.jsonl"
    if out.is_file() and out.stat().st_size > 0:
        return PromptJsonlDataset(
            path=out,
            input_key=dataset.input_key,
            label_key=dataset.label_key,
        )
    return dataset.to_prompt_jsonl(out_path=out)


# ---- shipping local datasets with the launch (ROADMAP 3.3) -----------------

INLINE_KIND = "inline_jsonl"
INLINE_MAX_BYTES = 16 * 1024 * 1024


def inline_dataset_blob(dataset: AnyDataset) -> dict[str, Any] | None:
    """Embed a *local* dataset in the launch payload; None = worker fetches it.

    Local JSONL files and local Harbor task folders do not exist on a BYO
    worker (it only has a git clone), so they travel with the job. HF and
    Harbor-registry datasets are downloaded on the worker instead.
    """
    import tempfile

    from daytona_gym.runtime.errors import DaytonaError, ErrorCode

    if isinstance(dataset, PromptJsonlDataset):
        path = dataset.resolved_path()
        if not path.is_file():
            return None  # a path that only exists on the worker — send as-is
        data, name = path.read_bytes(), path.name
    elif isinstance(dataset, HarborDataset) and not dataset.dataset_name:
        with tempfile.TemporaryDirectory() as tmp:
            out = dataset.to_prompt_jsonl(out_path=Path(tmp) / "harbor.jsonl")
            data = out.resolved_path().read_bytes()
        name = f"{dataset.cache_key() or 'harbor'}.jsonl"
    else:
        return None
    if len(data) > INLINE_MAX_BYTES:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"dataset {name} is {len(data) / 1e6:.1f} MB; launches embed local datasets up to "
            f"{INLINE_MAX_BYTES // (1024 * 1024)} MB. Use HuggingFaceDataset / a Harbor registry "
            "dataset, or place the file on the worker and pass its worker path.",
        )
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DaytonaError(ErrorCode.USER_CODE_ERROR, f"dataset {name} is not UTF-8 JSONL") from exc
    return {
        "kind": INLINE_KIND,
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "content": content,
        "input_key": dataset.input_key,
        "label_key": dataset.label_key,
    }


def materialize_inline(blob: dict[str, Any], *, runs_dir: Path) -> PromptJsonlDataset:
    """Worker side: write an embedded dataset to ``runs/data/`` after verifying it."""
    import re

    from daytona_gym.runtime.errors import DaytonaError, ErrorCode

    data = str(blob.get("content", "")).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if digest != blob.get("sha256"):
        raise DaytonaError(
            ErrorCode.PLATFORM_ERROR, "embedded dataset failed its checksum (payload corrupted in transit)"
        )
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(str(blob.get("name") or "dataset")).stem)[:64]
    out = Path(runs_dir) / "data" / f"inline_{digest[:12]}_{stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == digest):
        out.write_bytes(data)
    return PromptJsonlDataset(
        path=out,
        input_key=blob.get("input_key", "prompt"),
        label_key=blob.get("label_key", "label"),
    )
