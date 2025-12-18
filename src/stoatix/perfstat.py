"""Linux perf stat metrics collector.

This module provides functionality to run commands with perf stat
and parse the resulting hardware performance counters.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def is_supported() -> bool:
    """Check if perf stat collection is supported on this platform.

    Returns:
        True only on Linux systems.
    """
    return sys.platform == "linux"


def find_perf() -> str | None:
    """Find the perf executable on PATH.

    Returns:
        Path to perf executable, or None if not found.
    """
    return shutil.which("perf")


def run_with_perf_stat(
    cmd: list[str],
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout_s: float | None,
    events: list[str],
    out_file: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run a command wrapped with perf stat and collect metrics.

    Args:
        cmd: Command to execute as list of strings.
        cwd: Working directory for the command.
        env: Environment variables for the command.
        timeout_s: Timeout in seconds, or None for no timeout.
        events: List of perf events to collect (e.g., ["cycles", "instructions"]).
        out_file: Path where perf will write its CSV output.

    Returns:
        Tuple of (CompletedProcess, metrics_dict) where metrics_dict contains:
            - perf_ok: bool - Whether perf collection succeeded
            - perf_error: str|None - Error message if perf_ok is False
            - perf_events: list[str] - Events that were requested
            - perf_stat_path: str - Path to the perf output file
            - perf_stat: dict[str, float|None] - Parsed counter values
            - perf_raw: dict - Raw parsing metadata
    """
    metrics: dict[str, Any] = {
        "perf_ok": False,
        "perf_error": None,
        "perf_events": events,
        "perf_stat_path": str(out_file),
        "perf_stat": {},
        "perf_raw": {},
    }

    # Check platform support
    if not is_supported():
        metrics["perf_error"] = "perf stat is only supported on Linux"
        # Still run the command without perf
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=cwd,
                env=env,
            )
            return result, metrics
        except subprocess.TimeoutExpired as e:
            # Return a pseudo-CompletedProcess for timeout
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            )
            metrics["perf_error"] = f"Command timed out after {timeout_s}s"
            return result, metrics

    # Check perf availability
    perf_path = find_perf()
    if perf_path is None:
        metrics["perf_error"] = "perf executable not found on PATH"
        # Still run the command without perf
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=cwd,
                env=env,
            )
            return result, metrics
        except subprocess.TimeoutExpired as e:
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            )
            metrics["perf_error"] = f"Command timed out after {timeout_s}s"
            return result, metrics

    # Build perf stat command
    # -x, : CSV output with comma separator
    # -o file : Write stats to file instead of stderr
    # -e events : Events to collect
    # -- : Separator before command
    events_str = ",".join(events)
    perf_cmd = [
        "perf", "stat",
        "-x", ",",
        "-o", str(out_file),
        "-e", events_str,
        "--",
    ] + cmd

    metrics["perf_raw"]["perf_command"] = perf_cmd

    try:
        # Ensure output directory exists
        out_file.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            perf_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            env=env,
        )

        # Parse the perf output file
        if out_file.exists():
            perf_stat, parse_errors = _parse_perf_csv(out_file, events)
            metrics["perf_stat"] = perf_stat
            metrics["perf_raw"]["parse_errors"] = parse_errors

            if parse_errors:
                metrics["perf_raw"]["warnings"] = parse_errors

            # Consider perf OK if we got at least some metrics
            if perf_stat:
                metrics["perf_ok"] = True
            else:
                metrics["perf_error"] = "No metrics parsed from perf output"
        else:
            metrics["perf_error"] = f"perf output file not created: {out_file}"

        return result, metrics

    except subprocess.TimeoutExpired as e:
        result = subprocess.CompletedProcess(
            args=perf_cmd,
            returncode=-1,
            stdout=e.stdout or "" if hasattr(e, "stdout") else "",
            stderr=e.stderr or "" if hasattr(e, "stderr") else "",
        )
        metrics["perf_error"] = f"Command timed out after {timeout_s}s"

        # Try to parse partial output if file exists
        if out_file.exists():
            try:
                perf_stat, _ = _parse_perf_csv(out_file, events)
                metrics["perf_stat"] = perf_stat
                if perf_stat:
                    metrics["perf_ok"] = True
            except Exception:
                pass

        return result, metrics

    except FileNotFoundError:
        result = subprocess.CompletedProcess(
            args=perf_cmd,
            returncode=-1,
            stdout="",
            stderr=f"Command not found: {cmd[0] if cmd else 'empty command'}",
        )
        metrics["perf_error"] = f"Command not found: {cmd[0] if cmd else 'empty command'}"
        return result, metrics

    except OSError as e:
        result = subprocess.CompletedProcess(
            args=perf_cmd,
            returncode=-1,
            stdout="",
            stderr=str(e),
        )
        metrics["perf_error"] = f"OS error: {e}"
        return result, metrics

    except Exception as e:
        result = subprocess.CompletedProcess(
            args=perf_cmd,
            returncode=-1,
            stdout="",
            stderr=str(e),
        )
        metrics["perf_error"] = f"Unexpected error: {e}"
        return result, metrics


def _parse_perf_csv(
    path: Path,
    requested_events: list[str],
) -> tuple[dict[str, float | None], list[str]]:
    """Parse perf stat CSV output file.

    perf stat -x, output format varies but generally:
    value,unit,event-name,run-time,percent,optional-fields...

    Lines starting with # are comments/headers.
    Values may be:
    - Numeric (possibly with commas as thousands separators)
    - "<not supported>" or "<not counted>" for unavailable counters

    Args:
        path: Path to the perf output file.
        requested_events: List of events that were requested.

    Returns:
        Tuple of (parsed_stats, parse_errors) where:
            - parsed_stats: dict mapping event names to values (None if not available)
            - parse_errors: list of any parsing warnings/errors
    """
    stats: dict[str, float | None] = {}
    errors: list[str] = []

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {}, [f"Failed to read perf output file: {e}"]

    for line_num, line in enumerate(content.split("\n"), 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Parse CSV line
        parts = line.split(",")

        if len(parts) < 3:
            # Not enough fields for a valid perf stat line
            continue

        value_str = parts[0].strip()
        # parts[1] is typically unit (may be empty)
        event_name = parts[2].strip()

        if not event_name:
            continue

        # Parse the value
        value = _parse_perf_value(value_str)

        # Store the value keyed by event name
        stats[event_name] = value

    # Check for missing events
    for event in requested_events:
        if event not in stats:
            # Event might be reported with a different name (e.g., with :u suffix)
            # Try to find a partial match
            found = False
            for key in stats:
                if event in key or key in event:
                    found = True
                    break
            if not found:
                errors.append(f"Requested event '{event}' not found in perf output")

    return stats, errors


def _parse_perf_value(value_str: str) -> float | None:
    """Parse a perf stat value string to a float.

    Handles:
    - Regular numbers (possibly with comma thousands separators)
    - "<not supported>" -> None
    - "<not counted>" -> None
    - Empty strings -> None

    Args:
        value_str: The value string from perf output.

    Returns:
        Float value, or None if not available/parseable.
    """
    if not value_str:
        return None

    # Handle special perf messages
    lower = value_str.lower()
    if "<not supported>" in lower or "<not counted>" in lower:
        return None

    # Strip commas (thousands separators) and whitespace
    cleaned = value_str.replace(",", "").strip()

    if not cleaned:
        return None

    try:
        # Try to parse as float
        return float(cleaned)
    except ValueError:
        return None


def compute_derived(perf_stat: dict[str, float | None]) -> dict[str, float | None]:
    """Compute derived metrics from perf stat counters.

    Computes:
    - cpi: Cycles Per Instruction (cycles / instructions)
    - ipc: Instructions Per Cycle (instructions / cycles)
    - cache_miss_rate: cache-misses / cache-references

    Args:
        perf_stat: Dictionary of perf stat counter values.

    Returns:
        Dictionary of derived metrics (values are None if inputs missing or zero divisor).
    """
    derived: dict[str, float | None] = {
        "cpi": None,
        "ipc": None,
        "cache_miss_rate": None,
    }

    cycles = perf_stat.get("cycles")
    instructions = perf_stat.get("instructions")
    cache_refs = perf_stat.get("cache-references")
    cache_misses = perf_stat.get("cache-misses")

    # CPI = cycles / instructions
    if cycles is not None and instructions is not None and instructions != 0:
        derived["cpi"] = cycles / instructions

    # IPC = instructions / cycles
    if instructions is not None and cycles is not None and cycles != 0:
        derived["ipc"] = instructions / cycles

    # Cache miss rate = cache-misses / cache-references
    if cache_misses is not None and cache_refs is not None and cache_refs != 0:
        derived["cache_miss_rate"] = cache_misses / cache_refs

    return derived


# Default events to collect if none specified
DEFAULT_EVENTS = [
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
    "context-switches",
    "cpu-migrations",
    "page-faults",
]


def parse_perf_csv(
    path: Path,
    requested_events: list[str] | None = None,
) -> tuple[dict[str, float | None], list[str]]:
    """Parse perf stat CSV output file (public API for testing).

    perf stat -x, output format varies but generally:
    value,unit,event-name,run-time,percent,optional-fields...

    Lines starting with # are comments/headers.
    Values may be:
    - Numeric (possibly with commas as thousands separators)
    - "<not supported>" or "<not counted>" for unavailable counters

    Args:
        path: Path to the perf output file.
        requested_events: List of events that were requested (optional).

    Returns:
        Tuple of (parsed_stats, parse_errors) where:
            - parsed_stats: dict mapping event names to values (None if not available)
            - parse_errors: list of any parsing warnings/errors
    """
    return _parse_perf_csv(path, requested_events or [])
