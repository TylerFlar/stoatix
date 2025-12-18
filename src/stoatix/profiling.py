"""Deep profiling with Linux perf record and flamegraph generation.

This module provides functionality to:
1. Capture detailed profiling data using `perf record`
2. Generate flamegraph SVG visualizations from perf data

Requirements:
- Linux with `perf` installed for capture
- FlameGraph tools (stackcollapse-perf.pl, flamegraph.pl) for visualization
  - Either on PATH or in a specified directory

All functions are best-effort and return metadata about success/failure
rather than raising exceptions.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def is_linux() -> bool:
    """Check if the current platform is Linux.

    Returns:
        True if running on Linux, False otherwise.
    """
    return sys.platform == "linux"


def find_perf() -> str | None:
    """Find the perf executable on PATH.

    Returns:
        Path to perf executable, or None if not found.
    """
    return shutil.which("perf")


def find_flamegraph_tools(
    flamegraph_dir: Path | None = None,
) -> dict[str, str] | None:
    """Find FlameGraph tools (stackcollapse-perf.pl and flamegraph.pl).

    Searches in the following order:
    1. If flamegraph_dir is provided, look there first
    2. Fall back to PATH

    Args:
        flamegraph_dir: Optional directory containing FlameGraph tools.

    Returns:
        Dictionary with paths to tools if both found:
            {"stackcollapse": "/path/to/stackcollapse-perf.pl",
             "flamegraph": "/path/to/flamegraph.pl"}
        None if either tool is missing.
    """
    stackcollapse_path: str | None = None
    flamegraph_path: str | None = None

    # Check flamegraph_dir first if provided
    if flamegraph_dir is not None:
        flamegraph_dir = Path(flamegraph_dir)
        if flamegraph_dir.exists():
            sc_candidate = flamegraph_dir / "stackcollapse-perf.pl"
            fg_candidate = flamegraph_dir / "flamegraph.pl"

            if sc_candidate.exists() and sc_candidate.is_file():
                stackcollapse_path = str(sc_candidate)
            if fg_candidate.exists() and fg_candidate.is_file():
                flamegraph_path = str(fg_candidate)

    # Fall back to PATH for any missing tools
    if stackcollapse_path is None:
        stackcollapse_path = shutil.which("stackcollapse-perf.pl")
    if flamegraph_path is None:
        flamegraph_path = shutil.which("flamegraph.pl")

    # Both tools must be found
    if stackcollapse_path is None or flamegraph_path is None:
        return None

    return {
        "stackcollapse": stackcollapse_path,
        "flamegraph": flamegraph_path,
    }


def run_perf_record(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    perf_data_path: Path,
    freq: int = 99,
    call_graph: str = "dwarf",
) -> tuple[subprocess.CompletedProcess[str] | None, dict[str, Any]]:
    """Run a command under perf record to capture profiling data.

    Executes:
        perf record -F <freq> -g --call-graph <call_graph> -o <perf_data_path> -- <cmd...>

    Args:
        cmd: Command to profile as list of strings.
        cwd: Working directory for the command.
        env: Environment variables for the command.
        timeout_s: Timeout in seconds, or None for no timeout.
        perf_data_path: Path where perf will write its data file.
        freq: Sampling frequency in Hz (default 99).
        call_graph: Call graph recording method (default "dwarf").
            Options: "dwarf", "fp", "lbr".

    Returns:
        Tuple of (CompletedProcess or None, metadata dict) where metadata contains:
            - ok: bool - Whether perf record completed successfully
            - timed_out: bool - Whether the command timed out
            - exit_code: int|None - Exit code of perf record (None if not run)
            - elapsed_s: float - Wall clock time of the run
            - error: str|None - Error message if ok is False
            - perf_data_path: str - Path to the perf.data file
    """
    metadata: dict[str, Any] = {
        "ok": False,
        "timed_out": False,
        "exit_code": None,
        "elapsed_s": 0.0,
        "error": None,
        "perf_data_path": str(perf_data_path),
    }

    # Check platform
    if not is_linux():
        metadata["error"] = "perf record is only supported on Linux"
        return None, metadata

    # Check perf availability
    perf_path = find_perf()
    if perf_path is None:
        metadata["error"] = "perf executable not found on PATH"
        return None, metadata

    # Ensure output directory exists
    try:
        perf_data_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        metadata["error"] = f"Failed to create output directory: {e}"
        return None, metadata

    # Build perf record command
    perf_cmd = [
        "perf", "record",
        "-F", str(freq),
        "-g",
        "--call-graph", call_graph,
        "-o", str(perf_data_path),
        "--",
    ] + cmd

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            perf_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            env=env,
        )

        elapsed = time.perf_counter() - start_time
        metadata["elapsed_s"] = elapsed
        metadata["exit_code"] = result.returncode

        # Check if perf.data was created
        if perf_data_path.exists() and perf_data_path.stat().st_size > 0:
            metadata["ok"] = True
        else:
            metadata["error"] = "perf record did not create output file"
            if result.stderr:
                metadata["error"] += f": {result.stderr.strip()}"

        return result, metadata

    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start_time
        metadata["elapsed_s"] = elapsed
        metadata["timed_out"] = True
        metadata["error"] = f"Command timed out after {timeout_s}s"

        # Create a pseudo-CompletedProcess
        result = subprocess.CompletedProcess(
            args=perf_cmd,
            returncode=-1,
            stdout=e.stdout or "" if hasattr(e, "stdout") else "",
            stderr=e.stderr or "" if hasattr(e, "stderr") else "",
        )

        # perf.data might still have partial data
        if perf_data_path.exists() and perf_data_path.stat().st_size > 0:
            metadata["ok"] = True  # Partial data is still useful
            metadata["error"] = "Command timed out but partial data was captured"

        return result, metadata

    except FileNotFoundError as e:
        elapsed = time.perf_counter() - start_time
        metadata["elapsed_s"] = elapsed
        metadata["error"] = f"Command not found: {e}"
        return None, metadata

    except OSError as e:
        elapsed = time.perf_counter() - start_time
        metadata["elapsed_s"] = elapsed
        metadata["error"] = f"OS error: {e}"
        return None, metadata

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        metadata["elapsed_s"] = elapsed
        metadata["error"] = f"Unexpected error: {e}"
        return None, metadata


def generate_flamegraph(
    *,
    perf_data_path: Path,
    out_dir: Path,
    flamegraph_dir: Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate a flamegraph SVG from perf record data.

    Steps:
    1. Run `perf script -i perf_data_path` to generate perf.script
    2. Run `stackcollapse-perf.pl --all perf.script` to generate folded.txt
    3. Run `flamegraph.pl [--title ...] folded.txt` to generate flamegraph.svg

    Args:
        perf_data_path: Path to perf.data file from perf record.
        out_dir: Directory to write output files (perf.script, folded.txt, flamegraph.svg).
        flamegraph_dir: Optional directory containing FlameGraph tools.
        title: Optional title for the flamegraph.

    Returns:
        Dictionary with results:
            - flamegraph_ok: bool - Whether flamegraph was generated successfully
            - flamegraph_path: str|None - Relative path to flamegraph.svg
            - perf_script_path: str|None - Relative path to perf.script
            - folded_path: str|None - Relative path to folded.txt
            - error: str|None - Error message if flamegraph_ok is False
            - tools: dict|None - Paths to tools used
            - steps_completed: list[str] - List of successfully completed steps
    """
    result: dict[str, Any] = {
        "flamegraph_ok": False,
        "flamegraph_path": None,
        "perf_script_path": None,
        "folded_path": None,
        "error": None,
        "tools": None,
        "steps_completed": [],
    }

    # Define output paths (relative names for result, absolute for execution)
    perf_script_name = "perf.script"
    folded_name = "folded.txt"
    flamegraph_name = "flamegraph.svg"

    perf_script_path = out_dir / perf_script_name
    folded_path = out_dir / folded_name
    flamegraph_path = out_dir / flamegraph_name

    # Check platform
    if not is_linux():
        result["error"] = "Flamegraph generation requires Linux"
        return result

    # Check perf availability
    perf_path = find_perf()
    if perf_path is None:
        result["error"] = "perf executable not found on PATH"
        return result

    # Find FlameGraph tools
    tools = find_flamegraph_tools(flamegraph_dir)
    if tools is None:
        result["error"] = (
            "FlameGraph tools not found. "
            "Install stackcollapse-perf.pl and flamegraph.pl on PATH, "
            "or specify --flamegraph-dir"
        )
        return result

    result["tools"] = tools

    # Check input file
    if not perf_data_path.exists():
        result["error"] = f"perf data file not found: {perf_data_path}"
        return result

    # Ensure output directory exists
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result["error"] = f"Failed to create output directory: {e}"
        return result

    # Step 1: perf script
    try:
        perf_script_result = subprocess.run(
            ["perf", "script", "-i", str(perf_data_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large files
        )

        if perf_script_result.returncode != 0:
            result["error"] = f"perf script failed: {perf_script_result.stderr.strip()}"
            return result

        # Write perf.script
        perf_script_path.write_text(perf_script_result.stdout, encoding="utf-8")
        result["perf_script_path"] = perf_script_name
        result["steps_completed"].append("perf_script")

    except subprocess.TimeoutExpired:
        result["error"] = "perf script timed out (>5 minutes)"
        return result
    except Exception as e:
        result["error"] = f"perf script failed: {e}"
        return result

    # Step 2: stackcollapse-perf.pl
    try:
        # Use perl to run the script for cross-platform compatibility
        collapse_cmd = ["perl", tools["stackcollapse"], "--all", str(perf_script_path)]

        collapse_result = subprocess.run(
            collapse_cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if collapse_result.returncode != 0:
            result["error"] = f"stackcollapse-perf.pl failed: {collapse_result.stderr.strip()}"
            return result

        # Write folded.txt
        folded_path.write_text(collapse_result.stdout, encoding="utf-8")
        result["folded_path"] = folded_name
        result["steps_completed"].append("stackcollapse")

    except subprocess.TimeoutExpired:
        result["error"] = "stackcollapse-perf.pl timed out (>5 minutes)"
        return result
    except Exception as e:
        result["error"] = f"stackcollapse-perf.pl failed: {e}"
        return result

    # Step 3: flamegraph.pl
    try:
        fg_cmd = ["perl", tools["flamegraph"]]
        if title:
            fg_cmd.extend(["--title", title])
        fg_cmd.append(str(folded_path))

        fg_result = subprocess.run(
            fg_cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if fg_result.returncode != 0:
            result["error"] = f"flamegraph.pl failed: {fg_result.stderr.strip()}"
            return result

        # Write flamegraph.svg
        flamegraph_path.write_text(fg_result.stdout, encoding="utf-8")
        result["flamegraph_path"] = flamegraph_name
        result["steps_completed"].append("flamegraph")
        result["flamegraph_ok"] = True

    except subprocess.TimeoutExpired:
        result["error"] = "flamegraph.pl timed out (>2 minutes)"
        return result
    except Exception as e:
        result["error"] = f"flamegraph.pl failed: {e}"
        return result

    return result


def check_profiling_support(
    flamegraph_dir: Path | None = None,
) -> dict[str, Any]:
    """Check what profiling capabilities are available.

    Args:
        flamegraph_dir: Optional directory containing FlameGraph tools.

    Returns:
        Dictionary with availability information:
            - is_linux: bool
            - perf_available: bool
            - perf_path: str|None
            - flamegraph_tools_available: bool
            - flamegraph_tools: dict|None
            - can_record: bool - Can capture perf record data
            - can_flamegraph: bool - Can generate flamegraphs
            - issues: list[str] - List of issues preventing full support
    """
    result: dict[str, Any] = {
        "is_linux": is_linux(),
        "perf_available": False,
        "perf_path": None,
        "flamegraph_tools_available": False,
        "flamegraph_tools": None,
        "can_record": False,
        "can_flamegraph": False,
        "issues": [],
    }

    if not result["is_linux"]:
        result["issues"].append("Not running on Linux")
    else:
        perf_path = find_perf()
        if perf_path:
            result["perf_available"] = True
            result["perf_path"] = perf_path
            result["can_record"] = True
        else:
            result["issues"].append("perf executable not found on PATH")

        tools = find_flamegraph_tools(flamegraph_dir)
        if tools:
            result["flamegraph_tools_available"] = True
            result["flamegraph_tools"] = tools
            if result["perf_available"]:
                result["can_flamegraph"] = True
        else:
            result["issues"].append(
                "FlameGraph tools (stackcollapse-perf.pl, flamegraph.pl) not found"
            )

    return result
