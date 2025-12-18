"""System information gathering module."""

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import stoatix


def get_system_info() -> dict[str, Any]:
    """Gather comprehensive system information for audit-grade metadata.

    All probes are best-effort; failures return None for that field.

    Returns:
        Dictionary containing:
        - timestamp_utc: ISO-8601 UTC timestamp
        - os: name and version
        - kernel: release/version info
        - cpu: logical_cores and model
        - ram_bytes: total RAM
        - python: version, implementation, executable
        - stoatix_version: package version
        - hostname: machine hostname
        - argv: full command line
        - working_dir: current working directory
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": _get_os_info(),
        "kernel": _get_kernel_info(),
        "cpu": _get_cpu_info(),
        "ram_bytes": _get_ram_bytes(),
        "python": _get_python_info(),
        "stoatix_version": stoatix.__version__,
        "hostname": platform.node(),
        "argv": sys.argv,
        "working_dir": os.getcwd(),
    }


def _get_os_info() -> dict[str, str | None]:
    """Get OS name and version."""
    return {
        "name": platform.system(),
        "version": platform.version(),
    }


def _get_kernel_info() -> dict[str, str | None]:
    """Get kernel release and version from platform.uname()."""
    uname = platform.uname()
    return {
        "release": uname.release,
        "version": uname.version,
    }


def _get_cpu_info() -> dict[str, Any]:
    """Get CPU information including logical cores and model."""
    return {
        "logical_cores": os.cpu_count(),
        "model": _get_cpu_model(),
    }


def _get_cpu_model() -> str | None:
    """Get CPU model string (best-effort, platform-specific).

    - Linux: parse /proc/cpuinfo "model name"
    - macOS: sysctl -n machdep.cpu.brand_string
    - Windows: platform.processor() (limited info)
    """
    system = platform.system()

    try:
        if system == "Linux":
            return _get_cpu_model_linux()
        elif system == "Darwin":
            return _get_cpu_model_macos()
        elif system == "Windows":
            # platform.processor() gives limited info on Windows
            # but is the best we can do without external deps
            proc = platform.processor()
            return proc if proc else None
        else:
            return platform.processor() or None
    except Exception:
        return None


def _get_cpu_model_linux() -> str | None:
    """Parse /proc/cpuinfo for model name on Linux."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    # Format: "model name\t: Intel(R) Core(TM) ..."
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip()
        return None
    except (OSError, IOError):
        return None


def _get_cpu_model_macos() -> str | None:
    """Get CPU model via sysctl on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _get_ram_bytes() -> int | None:
    """Get total RAM in bytes (best-effort, platform-specific).

    - Linux: parse /proc/meminfo MemTotal
    - macOS: sysctl -n hw.memsize
    - Windows: ctypes GlobalMemoryStatusEx
    """
    system = platform.system()

    try:
        if system == "Linux":
            return _get_ram_bytes_linux()
        elif system == "Darwin":
            return _get_ram_bytes_macos()
        elif system == "Windows":
            return _get_ram_bytes_windows()
        else:
            return None
    except Exception:
        return None


def _get_ram_bytes_linux() -> int | None:
    """Parse /proc/meminfo for MemTotal on Linux."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:       16384000 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return kb * 1024  # Convert kB to bytes
        return None
    except (OSError, IOError, ValueError):
        return None


def _get_ram_bytes_macos() -> int | None:
    """Get RAM via sysctl on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        return None


def _get_ram_bytes_windows() -> int | None:
    """Get RAM via ctypes GlobalMemoryStatusEx on Windows."""
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys
        return None
    except (ImportError, AttributeError, OSError):
        return None


def _get_python_info() -> dict[str, str | None]:
    """Get Python version, implementation, and executable path."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
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
