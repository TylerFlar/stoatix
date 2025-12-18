"""System information gathering module."""

import os
import platform
import subprocess
from datetime import datetime, timezone


def get_system_info() -> dict[str, str | int | None]:
    """Gather system information.

    Returns:
        Dictionary containing system information.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }


def get_git_info(timeout: float = 5.0) -> dict[str, str | bool | None]:
    """Get git repository information.

    Attempts to gather git information from the current directory.
    Returns null fields if git is not available or not in a repository.

    Args:
        timeout: Timeout in seconds for git commands.

    Returns:
        Dictionary containing git information with keys:
        - commit: The current commit hash or None
        - branch: The current branch name or None
        - is_dirty: True if there are uncommitted changes, None if unknown
    """
    return {
        "commit": _git_command(["git", "rev-parse", "HEAD"], timeout),
        "branch": _git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout),
        "is_dirty": _git_is_dirty(timeout),
    }


def _git_command(args: list[str], timeout: float) -> str | None:
    """Run a git command and return its output.

    Args:
        args: Command and arguments to run.
        timeout: Timeout in seconds.

    Returns:
        Stripped stdout output, or None if the command fails.
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _git_is_dirty(timeout: float) -> bool | None:
    """Check if the git repository has uncommitted changes.

    Args:
        timeout: Timeout in seconds.

    Returns:
        True if dirty, False if clean, None if unable to determine.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=timeout,
        )
        # Exit code 0 means clean, non-zero means dirty
        return result.returncode != 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
