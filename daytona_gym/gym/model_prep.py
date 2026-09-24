"""Download HF weights + convert to Megatron torch_dist when missing on the GPU box."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from daytona_gym.gym.models import SoftSlimeModel
from daytona_gym.gym.slime_command import load_model_args
from daytona_gym.runtime.errors import DaytonaError, ErrorCode


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def _looks_like_hf_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "config.json").is_file() or any(path.glob("*.safetensors")) or any(
        path.glob("pytorch_model*.bin")
    )


def _looks_like_torch_dist(path: Path) -> bool:
    if not path.is_dir():
        return False
    # Megatron dist checkpoint layouts vary; non-empty dir after convert is enough.
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def ensure_model_ready(
    model: SoftSlimeModel,
    *,
    on_phase: Callable[[str, str], None] | None = None,
) -> None:
    """If HF / torch_dist dirs are missing, download + convert (uses ``HF_TOKEN``).

    No-ops unless this looks like a real Slime GPU box (convert script present),
    so CPU unit tests with stub paths still get a clean ``missing paths`` error.
    """
    def note(phase: str, message: str) -> None:
        if on_phase is not None:
            on_phase(phase, message)

    hf_dir = Path(os.environ.get("HF_CHECKPOINT", model.hf_checkpoint)).expanduser()
    ref_dir = Path(os.environ.get("REF_LOAD", model.ref_load)).expanduser()
    slime = Path(os.environ.get("SLIME_ROOT", model.slime_root)).expanduser()
    megatron = Path(os.environ.get("MEGATRON_ROOT", model.megatron_root)).expanduser()
    script = os.environ.get("MODEL_SCRIPT", model.model_script)
    repo = os.environ.get("HF_REPO", model.hf_repo)

    convert = slime / "tools" / "convert_hf_to_torch_dist.py"
    if not convert.is_file():
        return

    if not _looks_like_hf_checkpoint(hf_dir):
        note("model_download", f"Downloading {repo}…")
        _download_hf(repo=repo, dest=hf_dir)
    else:
        note("model_download", f"HF checkpoint ready at {hf_dir}")

    if not _looks_like_torch_dist(ref_dir):
        note("model_convert", f"Converting HF → torch_dist at {ref_dir}…")
        _convert_torch_dist(
            slime_root=slime,
            megatron_root=megatron,
            model_script=script,
            hf_checkpoint=hf_dir,
            save_dir=ref_dir,
        )
    else:
        note("model_convert", f"torch_dist ready at {ref_dir}")


def _download_hf(*, repo: str, dest: Path) -> None:
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    env = os.environ.copy()
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    print(f"model prep: downloading {repo} → {dest}", flush=True)
    if token:
        print("model prep: using HF_TOKEN from environment", flush=True)
    else:
        print(
            "model prep: no HF_TOKEN set — public download (slower / rate-limited)",
            flush=True,
        )

    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf_bin:
        cmd = [hf_bin]
        if Path(hf_bin).name == "huggingface-cli":
            cmd.append("download")
        else:
            cmd.append("download")
        cmd.extend([repo, "--local-dir", str(dest)])
        if token:
            cmd.extend(["--token", token])
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                f"hf download failed for {repo} (exit {proc.returncode})",
            )
    else:
        # Fallback: huggingface_hub Python API
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DaytonaError(
                ErrorCode.USER_CODE_ERROR,
                "hf/huggingface-cli not on PATH and huggingface_hub not installed; "
                "cannot download model weights",
            ) from exc
        snapshot_download(
            repo_id=repo,
            local_dir=str(dest),
            token=token,
        )

    if not _looks_like_hf_checkpoint(dest):
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"download finished but {dest} does not look like an HF checkpoint",
        )


def _convert_torch_dist(
    *,
    slime_root: Path,
    megatron_root: Path,
    model_script: str,
    hf_checkpoint: Path,
    save_dir: Path,
) -> None:
    convert = slime_root / "tools" / "convert_hf_to_torch_dist.py"
    if not convert.is_file():
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"Slime convert script not found: {convert}",
        )
    if not megatron_root.is_dir():
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"Megatron root missing: {megatron_root}",
        )

    save_dir = save_dir.resolve()
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    model_args = load_model_args(slime_root, model_script)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{megatron_root}:{py_path}" if py_path else str(megatron_root)
    )

    cmd = [
        "python",
        str(convert),
        *model_args,
        "--hf-checkpoint",
        str(hf_checkpoint.resolve()),
        "--save",
        str(save_dir),
    ]
    print(f"model prep: converting HF → torch_dist at {save_dir}", flush=True)
    proc = subprocess.run(cmd, cwd=str(slime_root), env=env, check=False)
    if proc.returncode != 0:
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"convert_hf_to_torch_dist failed (exit {proc.returncode})",
        )
    if not _looks_like_torch_dist(save_dir):
        raise DaytonaError(
            ErrorCode.USER_CODE_ERROR,
            f"convert finished but {save_dir} is empty",
        )
