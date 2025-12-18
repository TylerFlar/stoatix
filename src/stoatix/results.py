"""Benchmark results handling."""

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from stoatix.plan import CaseSpec


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


def write_session_metadata(
    out_dir: Path,
    *,
    suite_id: str,
    config_path: str,
    config_hash: str,
    benchmark_count: int,
    case_count: int,
    shuffle_enabled: bool = False,
    seed: int | None = None,
    system_info: dict[str, Any],
    git_info: dict[str, Any],
) -> None:
    """Write session metadata to session.json in the output directory.

    Args:
        out_dir: Output directory path.
        suite_id: Unique identifier for this suite run.
        config_path: Path to the configuration file.
        config_hash: Hash of the configuration file.
        benchmark_count: Number of benchmarks in the suite.
        case_count: Number of cases after matrix expansion.
        shuffle_enabled: Whether case order was shuffled.
        seed: Random seed used for shuffling (None if not shuffled).
        system_info: System information dictionary.
        git_info: Git repository information dictionary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "suite_id": suite_id,
        "config_path": config_path,
        "config_hash": config_hash,
        "benchmark_count": benchmark_count,
        "case_count": case_count,
        "shuffle_enabled": shuffle_enabled,
        "seed": seed,
        "system_info": system_info,
        "git_info": git_info,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    session_path = out_dir / "session.json"
    with session_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def write_cases(
    out_dir: Path,
    cases: list[CaseSpec],
    *,
    suite_id: str,
) -> None:
    """Write expanded cases to cases.json in the output directory.

    Args:
        out_dir: Output directory path.
        cases: List of CaseSpec objects to write.
        suite_id: Unique identifier for this suite run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_data: list[dict[str, Any]] = []
    for case in cases:
        cases_data.append({
            "bench_name": case.bench_name,
            "case_id": case.case_id,
            "case_key": case.case_key,
            "params": case.params,
            "command": case.command,
            "cwd": case.cwd,
            "env": case.env,
            "warmups": case.warmups,
            "runs": case.runs,
            "retries": case.retries,
            "timeout_s": case.timeout_s,
            "pin": {
                "strategy": case.pin.strategy,
                "cores": list(case.pin.cores) if case.pin.cores else [],
            },
        })

    output: dict[str, Any] = {
        "suite_id": suite_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases_data,
    }

    cases_path = out_dir / "cases.json"
    with cases_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def write_resolved_config(out_dir: Path, resolved: dict[str, Any]) -> None:
    """Write resolved configuration to config.resolved.yml.

    The resolved config has all defaults explicitly filled in
    and each benchmark shows inherited + overridden values.

    Args:
        out_dir: Output directory path.
        resolved: Resolved configuration dictionary from SuiteConfig.to_resolved_dict().
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.resolved.yml"
    with config_path.open("w", encoding="utf-8") as f:
        # Add a header comment
        f.write("# Resolved configuration with all defaults expanded\n")
        f.write(f"# Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
        yaml.safe_dump(resolved, f, default_flow_style=False, sort_keys=False)


def generate_suite_id() -> str:
    """Generate a new unique suite ID.

    Returns:
        UUID4 string.
    """
    return str(uuid4())
