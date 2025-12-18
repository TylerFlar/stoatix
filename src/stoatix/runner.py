"""Benchmark runner module."""

import hashlib
import logging
import os
import random
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from stoatix.config import load_config
from stoatix.plan import CaseSpec, expand_suite
from stoatix.results import (
    generate_suite_id,
    write_cases,
    write_jsonl,
    write_resolved_config,
    write_session_metadata,
)
from stoatix.sysinfo import get_git_info, get_system_info

logger = logging.getLogger(__name__)

# Maximum characters to keep for stdout/stderr truncation (head + tail)
MAX_OUTPUT_CHARS = 4096
TRUNC_HEAD_CHARS = 2048
TRUNC_TAIL_CHARS = 2048

# Track if perf warning has been emitted (once per run)
_perf_warning_emitted = False


def run_suite(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    perf_stat: bool = False,
    perf_events: str = "cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults",
    perf_strict: bool = False,
) -> None:
    """Run the benchmark suite based on the provided configuration.

    Args:
        config_path: Path to the YAML configuration file.
        out_dir: Output directory for results.
        shuffle: Whether to shuffle case execution order.
        seed: Random seed for shuffling. If shuffle=True and seed=None,
              a random seed will be generated.
        perf_stat: Whether to collect Linux perf stat counters.
        perf_events: Comma-separated list of perf events to collect.
        perf_strict: If True, fail the run if perf stat cannot be collected.
    """
    config_path = Path(config_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and hash config
    config = load_config(config_path)
    config_hash = _compute_file_hash(config_path)

    suite_id = generate_suite_id()

    logger.info("Starting benchmark suite from: %s", config_path)
    logger.info("Suite ID: %s", suite_id)
    logger.info("Config hash: %s", config_hash[:16])
    logger.debug("Output directory: %s", out_path)

    # Expand config to cases
    cases = expand_suite(config)
    logger.info("Expanded to %d case(s)", len(cases))

    # Handle shuffle
    actual_seed: int | None = None
    if shuffle:
        if seed is None:
            actual_seed = secrets.randbits(32)
        else:
            actual_seed = seed
        logger.info("Shuffling cases with seed: %d", actual_seed)
        rng = random.Random(actual_seed)
        rng.shuffle(cases)

    # Write session metadata with comprehensive info
    write_session_metadata(
        out_path,
        suite_id=suite_id,
        config_path=str(config_path),
        config_hash=config_hash,
        benchmark_count=len(config.benchmarks),
        case_count=len(cases),
        shuffle_enabled=shuffle,
        seed=actual_seed,
        system_info=get_system_info(),
        git_info=get_git_info(),
    )

    # Write resolved config with all defaults expanded
    write_resolved_config(out_path, config.to_resolved_dict())

    # Write cases manifest (in execution order after potential shuffle)
    write_cases(out_path, cases, suite_id=suite_id)

    # Check perf availability if requested
    perf_available = False
    perf_error_reason: str | None = None

    if perf_stat:
        from stoatix.perfstat import find_perf, is_supported

        if not is_supported():
            perf_error_reason = "perf stat is only supported on Linux"
        elif find_perf() is None:
            perf_error_reason = "perf executable not found on PATH"
        else:
            # Quick check that perf can actually collect stats
            perf_available = _check_perf_available()
            if not perf_available:
                perf_error_reason = (
                    "perf stat check failed (insufficient permissions or kernel config)"
                )

        if not perf_available:
            if perf_strict:
                raise RuntimeError(
                    f"perf stat requested with --perf-strict but: {perf_error_reason}. "
                    "Ensure 'perf' is installed and you have permissions to use it."
                )
            else:
                logger.warning(
                    "perf stat requested but: %s. Continuing with timing-only measurements.",
                    perf_error_reason,
                )

    # Parse perf events list
    perf_events_list = [e.strip() for e in perf_events.split(",") if e.strip()]

    # Reset per-run warning flag
    global _perf_warning_emitted
    _perf_warning_emitted = False

    # Run cases
    _run_cases(
        cases,
        suite_id,
        out_path,
        perf_stat=perf_stat,
        perf_available=perf_available,
        perf_events_list=perf_events_list,
        perf_error_reason=perf_error_reason,
    )

    logger.info("Results saved to: %s", out_path)


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def _truncate_output(text: str) -> str:
    """Truncate output using head+tail strategy.

    If text is longer than MAX_OUTPUT_CHARS, keep the first TRUNC_HEAD_CHARS
    and last TRUNC_TAIL_CHARS characters with a marker in between.

    Args:
        text: Original output text.

    Returns:
        Truncated text.
    """
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    head = text[:TRUNC_HEAD_CHARS]
    tail = text[-TRUNC_TAIL_CHARS:]
    omitted = len(text) - TRUNC_HEAD_CHARS - TRUNC_TAIL_CHARS
    return f"{head}\n... [{omitted} chars omitted] ...\n{tail}"


def _check_perf_available() -> bool:
    """Check if perf stat is available on the system.

    Returns:
        True if perf is available and can collect stats, False otherwise.
    """
    import sys

    if sys.platform != "linux":
        logger.debug("perf stat is only available on Linux")
        return False

    try:
        result = subprocess.run(
            ["perf", "stat", "-e", "cycles", "--", "true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # perf stat outputs to stderr, check if we got any output
        if result.returncode == 0 or "cycles" in result.stderr.lower():
            logger.debug("perf stat is available")
            return True
        logger.debug("perf stat not available: %s", result.stderr)
        return False
    except FileNotFoundError:
        logger.debug("perf command not found")
        return False
    except subprocess.TimeoutExpired:
        logger.debug("perf stat check timed out")
        return False
    except Exception as e:
        logger.debug("perf stat check failed: %s", e)
        return False


def _run_cases(
    cases: list[CaseSpec],
    suite_id: str,
    out_dir: Path,
    *,
    perf_stat: bool = False,
    perf_available: bool = False,
    perf_events_list: list[str] | None = None,
    perf_error_reason: str | None = None,
) -> None:
    """Run all expanded cases.

    Args:
        cases: List of CaseSpec objects to run.
        suite_id: Unique identifier for this suite run.
        out_dir: Output directory for results.
        perf_stat: Whether perf stat collection was requested.
        perf_available: Whether perf stat is actually available.
        perf_events_list: List of perf events to collect.
        perf_error_reason: Reason why perf is unavailable (if applicable).
    """
    results_file = out_dir / "results.jsonl"
    perf_stat_dir = out_dir / "perf_stat"

    if not cases:
        logger.warning("No cases to run.")
        return

    # Create perf_stat directory if we'll be collecting perf data
    if perf_stat and perf_available:
        perf_stat_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_label = f"{case.bench_name}"
        if case.case_key:
            case_label += f" [{case.case_key}]"

        logger.info("Running case: %s", case_label)
        logger.debug(
            "  command=%s, warmups=%d, runs=%d, retries=%d, timeout_s=%s",
            case.command,
            case.warmups,
            case.runs,
            case.retries,
            case.timeout_s,
        )
        logger.debug("  case_id=%s", case.case_id)

        # Run warmups
        for i in range(case.warmups):
            logger.debug("  Warmup %d/%d", i + 1, case.warmups)
            records = _execute_run_with_retries(
                suite_id=suite_id,
                case=case,
                run_kind="warmup",
                iteration=i,
                out_dir=out_dir,
                perf_stat=perf_stat,
                perf_available=perf_available,
                perf_events_list=perf_events_list or [],
                perf_error_reason=perf_error_reason,
            )
            write_jsonl(results_file, records)

        # Run measured iterations
        for i in range(case.runs):
            logger.debug("  Run %d/%d", i + 1, case.runs)
            records = _execute_run_with_retries(
                suite_id=suite_id,
                case=case,
                run_kind="measured",
                iteration=i,
                out_dir=out_dir,
                perf_stat=perf_stat,
                perf_available=perf_available,
                perf_events_list=perf_events_list or [],
                perf_error_reason=perf_error_reason,
            )
            write_jsonl(results_file, records)

        logger.info("Case %s completed", case_label)


def _execute_run_with_retries(
    suite_id: str,
    case: CaseSpec,
    run_kind: Literal["warmup", "measured"],
    iteration: int,
    *,
    out_dir: Path,
    perf_stat: bool = False,
    perf_available: bool = False,
    perf_events_list: list[str] | None = None,
    perf_error_reason: str | None = None,
) -> list[dict[str, Any]]:
    """Execute a single benchmark run with retry support.

    Retries on timeout or non-zero exit code, up to (1 + retries) attempts.
    Each attempt produces a record.

    Args:
        suite_id: Unique identifier for this suite run.
        case: CaseSpec to execute.
        run_kind: Type of run - "warmup" or "measured".
        iteration: Iteration number (0-indexed).
        out_dir: Output directory for results.
        perf_stat: Whether perf stat collection was requested.
        perf_available: Whether perf stat is actually available.
        perf_events_list: List of perf events to collect.
        perf_error_reason: Reason why perf is unavailable (if applicable).

    Returns:
        List of run record dictionaries (one per attempt).
    """
    max_attempts = 1 + case.retries
    records: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        record = _execute_single_attempt(
            suite_id=suite_id,
            case=case,
            run_kind=run_kind,
            iteration=iteration,
            attempt=attempt,
            out_dir=out_dir,
            perf_stat=perf_stat,
            perf_available=perf_available,
            perf_events_list=perf_events_list or [],
            perf_error_reason=perf_error_reason,
        )
        records.append(record)

        # Check if we should retry
        if record["ok"]:
            # Success, no need to retry
            break

        if attempt < max_attempts:
            # Will retry
            logger.debug(
                "    Attempt %d failed, retrying (%d/%d)",
                attempt,
                attempt + 1,
                max_attempts,
            )

    return records


def _execute_single_attempt(
    suite_id: str,
    case: CaseSpec,
    run_kind: Literal["warmup", "measured"],
    iteration: int,
    attempt: int,
    *,
    out_dir: Path,
    perf_stat: bool = False,
    perf_available: bool = False,
    perf_events_list: list[str] | None = None,
    perf_error_reason: str | None = None,
) -> dict[str, Any]:
    """Execute a single attempt of a benchmark run.

    Args:
        suite_id: Unique identifier for this suite run.
        case: CaseSpec to execute.
        run_kind: Type of run - "warmup" or "measured".
        iteration: Iteration number (0-indexed).
        attempt: Attempt number (1-based).
        out_dir: Output directory for results.
        perf_stat: Whether perf stat collection was requested.
        perf_available: Whether perf stat is actually available.
        perf_events_list: List of perf events to collect.
        perf_error_reason: Reason why perf is unavailable (if applicable).

    Returns:
        Run record dictionary.
    """
    global _perf_warning_emitted

    elapsed_s: float = 0.0
    exit_code: int | None = None
    ok: bool = False
    timed_out: bool = False
    stdout_text: str = ""
    stderr_text: str = ""

    # Initialize metrics dict (always present when perf_stat requested)
    metrics: dict[str, Any] | None = None
    if perf_stat:
        metrics = {
            "perf_stat": {},
            "derived": {"cpi": None, "ipc": None, "cache_miss_rate": None},
            "perf_stat_path": None,
            "perf_ok": False,
            "perf_error": perf_error_reason,
            "perf_events": perf_events_list or [],
        }

    # Prepare environment (inherit current env + case-specific overrides)
    env = os.environ.copy()
    env.update(case.env)

    # Prepare working directory
    cwd = case.cwd

    started_at = datetime.now(timezone.utc)

    # Determine if we should use perf stat for this attempt
    use_perf = perf_stat and perf_available and perf_events_list

    if use_perf:
        # Use perfstat module
        from stoatix.perfstat import compute_derived, run_with_perf_stat

        # Create unique output path for this attempt
        perf_out_file = (
            out_dir
            / "perf_stat"
            / case.case_id
            / f"{run_kind}_iter{iteration}_attempt{attempt}.csv"
        )
        perf_out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            start = time.perf_counter()
            result, perf_metrics = run_with_perf_stat(
                cmd=case.command,
                cwd=cwd,
                env=env,
                timeout_s=case.timeout_s,
                events=perf_events_list,
                out_file=perf_out_file,
            )
            elapsed_s = time.perf_counter() - start

            exit_code = result.returncode
            ok = exit_code == 0
            stdout_text = result.stdout or ""
            stderr_text = result.stderr or ""

            # Update metrics from perfstat module
            if metrics is not None:
                # Compute relative path for portability
                rel_perf_path = perf_out_file.relative_to(out_dir)
                metrics["perf_stat"] = perf_metrics.get("perf_stat", {})
                metrics["perf_stat_path"] = str(rel_perf_path)
                metrics["perf_ok"] = perf_metrics.get("perf_ok", False)
                metrics["perf_error"] = perf_metrics.get("perf_error")
                metrics["perf_events"] = perf_metrics.get("perf_events", [])

                # Compute derived metrics
                if metrics["perf_ok"] and metrics["perf_stat"]:
                    metrics["derived"] = compute_derived(metrics["perf_stat"])

            if not ok:
                logger.warning(
                    "    %s iteration %d attempt %d failed with exit code %d",
                    run_kind,
                    iteration,
                    attempt,
                    exit_code,
                )

        except subprocess.TimeoutExpired as e:
            elapsed_s = case.timeout_s or 0.0
            exit_code = None
            ok = False
            timed_out = True
            stdout_text = e.stdout or "" if hasattr(e, "stdout") and e.stdout else ""
            stderr_text = e.stderr or "" if hasattr(e, "stderr") and e.stderr else ""
            if metrics is not None:
                metrics["perf_error"] = f"Command timed out after {case.timeout_s}s"
            logger.warning(
                "    %s iteration %d attempt %d timed out after %ss",
                run_kind,
                iteration,
                attempt,
                case.timeout_s,
            )

        except FileNotFoundError:
            exit_code = None
            ok = False
            stderr_text = f"Command not found: {case.command[0]}"
            if metrics is not None:
                metrics["perf_error"] = f"Command not found: {case.command[0]}"
            logger.error("    Command not found: %s", case.command[0])

        except OSError as e:
            exit_code = None
            ok = False
            stderr_text = str(e)
            if metrics is not None:
                metrics["perf_error"] = f"OS error: {e}"
            logger.error("    OS error: %s", e)

    else:
        # Run without perf stat
        if perf_stat and not perf_available and not _perf_warning_emitted:
            # Emit warning once per run
            logger.warning(
                "perf stat unavailable (%s), running without hardware counters",
                perf_error_reason or "unknown reason",
            )
            _perf_warning_emitted = True

        try:
            start = time.perf_counter()
            result = subprocess.run(
                case.command,
                capture_output=True,
                text=True,
                timeout=case.timeout_s,
                cwd=cwd,
                env=env,
            )
            elapsed_s = time.perf_counter() - start
            exit_code = result.returncode
            ok = exit_code == 0
            stdout_text = result.stdout or ""
            stderr_text = result.stderr or ""

            if not ok:
                logger.warning(
                    "    %s iteration %d attempt %d failed with exit code %d",
                    run_kind,
                    iteration,
                    attempt,
                    exit_code,
                )

        except subprocess.TimeoutExpired as e:
            elapsed_s = case.timeout_s or 0.0
            exit_code = None
            ok = False
            timed_out = True
            stdout_text = e.stdout or "" if hasattr(e, "stdout") and e.stdout else ""
            stderr_text = e.stderr or "" if hasattr(e, "stderr") and e.stderr else ""
            logger.warning(
                "    %s iteration %d attempt %d timed out after %ss",
                run_kind,
                iteration,
                attempt,
                case.timeout_s,
            )

        except FileNotFoundError:
            exit_code = None
            ok = False
            stderr_text = f"Command not found: {case.command[0]}"
            logger.error("    Command not found: %s", case.command[0])

        except OSError as e:
            exit_code = None
            ok = False
            stderr_text = str(e)
            logger.error("    OS error: %s", e)

    ended_at = datetime.now(timezone.utc)

    # Build record with full schema
    record: dict[str, Any] = {
        "suite_id": suite_id,
        "bench_name": case.bench_name,
        "case_id": case.case_id,
        "case_key": case.case_key,
        "params": case.params,
        "command": case.command,
        "cwd": case.cwd,
        "env_overrides": case.env,
        "run_kind": run_kind,
        "iteration": iteration,
        "attempt": attempt,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "elapsed_s": elapsed_s,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "ok": ok,
        "stdout_len": len(stdout_text),
        "stderr_len": len(stderr_text),
        "stdout_trunc": _truncate_output(stdout_text),
        "stderr_trunc": _truncate_output(stderr_text),
    }

    # Add metrics if perf_stat was requested (even if unavailable)
    if metrics is not None:
        record["metrics"] = metrics

    return record
