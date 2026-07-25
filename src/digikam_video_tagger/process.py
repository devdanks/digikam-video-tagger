from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


class CommandError(RuntimeError):
    """Raised when an external tool exits unsuccessfully."""


def run_command(
    args: Iterable[str | Path],
    *,
    timeout: float | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(arg) for arg in args]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
        timeout=timeout,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW")
        else 0,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if isinstance(completed.stderr, str) else ""
        raise CommandError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n{stderr}"
        )
    return completed
