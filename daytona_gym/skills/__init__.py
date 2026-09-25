"""Install the Daytona Gym agent skill bundle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_BUNDLE = Path(__file__).resolve().parent / "bundle"


def install_skills(dest: Path | None = None) -> Path:
    """Copy skill markdown into ``.cursor/skills/daytona-gym/`` (or ``dest``)."""
    target = Path(dest or Path.cwd() / ".cursor" / "skills" / "daytona-gym")
    target.mkdir(parents=True, exist_ok=True)
    if not _BUNDLE.is_dir():
        raise FileNotFoundError(f"skill bundle missing: {_BUNDLE}")
    for path in _BUNDLE.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    agents = Path.cwd() / "AGENTS.md"
    snippet = (
        "\n\n## Daytona Gym\n\n"
        "See `.cursor/skills/daytona-gym/SKILL.md` for TrainConfig + "
        "`dg run` / `dg dash` happy path.\n"
    )
    if agents.is_file():
        text = agents.read_text(encoding="utf-8")
        if "daytona-gym/SKILL.md" not in text:
            agents.write_text(text.rstrip() + snippet, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("dg skills install [--dest DIR]\n")
        return 0
    if args[0] != "install":
        print(f"unknown skills command: {args[0]}", file=sys.stderr)
        return 2
    dest = None
    if "--dest" in args:
        idx = args.index("--dest")
        if idx + 1 < len(args):
            dest = Path(args[idx + 1])
    path = install_skills(dest)
    print(f"installed skills → {path}")
    return 0
