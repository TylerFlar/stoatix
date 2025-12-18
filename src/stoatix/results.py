"""Benchmark results handling."""

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to a JSONL file (append-safe).

    Each record is written as a single JSON line. The file is opened
    in append mode, making it safe to call multiple times.

    Args:
        path: Path to the JSONL file.
        records: Iterable of dictionaries to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_session_metadata(out_dir: Path, metadata: dict[str, Any]) -> None:
    """Write session metadata to session.json in the output directory.

    Args:
        out_dir: Output directory path.
        metadata: Metadata dictionary to write.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    session_path = out_dir / "session.json"
    with session_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def new_run_record(
    suite_id: str,
    bench_name: str,
    command: list[str],
    run_kind: Literal["warmup", "measured"],
    iteration: int,
    elapsed_s: float,
    exit_code: int,
    ok: bool,
) -> dict[str, Any]:
    """Create a new run record dictionary.

    Args:
        suite_id: Unique identifier for the benchmark suite run (uuid4 string).
        bench_name: Name of the benchmark.
        command: Command that was executed (list of strings).
        run_kind: Type of run - "warmup" or "measured".
        iteration: Iteration number (0-indexed).
        elapsed_s: Elapsed time in seconds.
        exit_code: Process exit code.
        ok: Whether the run was successful.

    Returns:
        Dictionary containing the run record.
    """
    return {
        "suite_id": suite_id,
        "bench_name": bench_name,
        "command": command,
        "run_kind": run_kind,
        "iteration": iteration,
        "elapsed_s": elapsed_s,
        "exit_code": exit_code,
        "ok": ok,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_suite_id() -> str:
    """Generate a new unique suite ID.

    Returns:
        UUID4 string.
    """
    return str(uuid4())
