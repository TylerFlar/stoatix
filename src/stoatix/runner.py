"""Benchmark runner module."""

import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

from stoatix.config import load_config
from stoatix.plan import CaseSpec, expand_suite
from stoatix.results import (
    generate_suite_id,
    new_run_record,
    write_jsonl,
    write_session_metadata,
)
from stoatix.sysinfo import get_git_info, get_system_info

logger = logging.getLogger(__name__)


def run_suite(config_path: str | Path, out_dir: str | Path) -> None:
    """Run the benchmark suite based on the provided configuration.

    Args:
        config_path: Path to the YAML configuration file.
        out_dir: Output directory for results.
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

    # Collect and write session metadata
    metadata = {
        "suite_id": suite_id,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "system_info": get_system_info(),
        "git_info": get_git_info(),
    }
    write_session_metadata(out_path, metadata)

    # Expand config to cases and run
    cases = expand_suite(config)
    logger.info("Expanded to %d case(s)", len(cases))

    _run_cases(cases, suite_id, out_path)

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


def _run_cases(cases: list[CaseSpec], suite_id: str, out_dir: Path) -> None:
    """Run all expanded cases.

    Args:
        cases: List of CaseSpec objects to run.
        suite_id: Unique identifier for this suite run.
        out_dir: Output directory for results.
    """
    runs_file = out_dir / "runs.jsonl"

    if not cases:
        logger.warning("No cases to run.")
        return

    for case in cases:
        case_label = f"{case.bench_name}"
        if case.case_key:
            case_label += f" [{case.case_key}]"

        logger.info("Running case: %s", case_label)
        logger.debug(
            "  command=%s, warmups=%d, runs=%d, timeout_s=%s",
            case.command,
            case.warmups,
            case.runs,
            case.timeout_s,
        )
        logger.debug("  case_id=%s", case.case_id)

        # Run warmups
        for i in range(case.warmups):
            logger.debug("  Warmup %d/%d", i + 1, case.warmups)
            record = _execute_run(
                suite_id=suite_id,
                case=case,
                run_kind="warmup",
                iteration=i,
            )
            write_jsonl(runs_file, [record])

        # Run measured iterations
        for i in range(case.runs):
            logger.debug("  Run %d/%d", i + 1, case.runs)
            record = _execute_run(
                suite_id=suite_id,
                case=case,
                run_kind="measured",
                iteration=i,
            )
            write_jsonl(runs_file, [record])

        logger.info("Case %s completed", case_label)


def _execute_run(
    suite_id: str,
    case: CaseSpec,
    run_kind: Literal["warmup", "measured"],
    iteration: int,
) -> dict[str, Any]:
    """Execute a single benchmark run.

    Args:
        suite_id: Unique identifier for this suite run.
        case: CaseSpec to execute.
        run_kind: Type of run - "warmup" or "measured".
        iteration: Iteration number (0-indexed).

    Returns:
        Run record dictionary.
    """
    elapsed_s: float = 0.0
    exit_code: int = -1
    ok: bool = False
    error_msg: str | None = None

    # Prepare environment (inherit current env + case-specific overrides)
    env = os.environ.copy()
    env.update(case.env)

    # Prepare working directory
    cwd = case.cwd

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

        if not ok:
            error_msg = result.stderr.strip() or result.stdout.strip() or None
            logger.warning(
                "  %s iteration %d failed with exit code %d",
                run_kind,
                iteration,
                exit_code,
            )

    except subprocess.TimeoutExpired:
        elapsed_s = case.timeout_s or 0.0
        exit_code = -1
        ok = False
        error_msg = f"Timeout after {case.timeout_s}s"
        logger.warning(
            "  %s iteration %d timed out after %ss",
            run_kind,
            iteration,
            case.timeout_s,
        )

    except FileNotFoundError:
        exit_code = -1
        ok = False
        error_msg = f"Command not found: {case.command[0]}"
        logger.error("  Command not found: %s", case.command[0])

    except OSError as e:
        exit_code = -1
        ok = False
        error_msg = str(e)
        logger.error("  OS error: %s", e)

    record = new_run_record(
        suite_id=suite_id,
        bench_name=case.bench_name,
        command=case.command,
        run_kind=run_kind,
        iteration=iteration,
        elapsed_s=elapsed_s,
        exit_code=exit_code,
        ok=ok,
    )

    # Add case-specific fields
    record["case_id"] = case.case_id
    record["case_key"] = case.case_key
    record["params"] = case.params
    record["cwd"] = case.cwd
    record["env_overrides"] = case.env

    if error_msg:
        record["error"] = error_msg

    return record
