"""PtyShell echoes the remote command — marker parsing must ignore that echo."""

from __future__ import annotations

import base64
import json

from daytona_gym.telemetry.dashboard_sync import (
    _b64_section_after_label,
    _slice_marked_block,
)


def test_slice_marked_block_ignores_command_echo() -> None:
    # Realistic PtyShell buffer: command line contains marker strings mid-line,
    # then the shell prints the real delimiters on their own lines.
    payload = {"phase": "boot", "message": "Worker process started"}
    b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    text = (
        "printf '%s\\n' __DG_LIVE_BEGIN__; printf '%s\\n' __PROGRESS_B64__; "
        f"(base64 /tmp/x.json | tr -d '\\n'); printf '\\n%s\\n' __DG_LIVE_END__\n"
        "__DG_LIVE_BEGIN__\n"
        "__PROGRESS_B64__\n"
        f"{b64}\n"
        "__DG_LIVE_END__\n"
        "root@host:~# \n"
    )
    body = _slice_marked_block(text, "__DG_LIVE_BEGIN__", "__DG_LIVE_END__")
    assert "__DG_LIVE_BEGIN__" not in body
    assert "__DG_LIVE_END__" not in body
    assert "printf" not in body
    prog_b64 = _b64_section_after_label(body, "__PROGRESS_B64__", None)
    assert json.loads(base64.b64decode(prog_b64)) == payload


def test_naive_split_would_fail_but_line_slice_works() -> None:
    """Document the bug: str.split grabs the mid-echo gap."""
    text = (
        "echo __DG_BEGIN__; echo HI; echo __DG_END__\n"
        "__DG_BEGIN__\n"
        "REAL\n"
        "__DG_END__\n"
    )
    naive = text.split("__DG_BEGIN__", 1)[1].split("__DG_END__", 1)[0]
    assert "REAL" not in naive  # empty / junk from command echo
    body = _slice_marked_block(text, "__DG_BEGIN__", "__DG_END__")
    assert body.strip() == "REAL"
