"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    command: list[str]
    warmups: int = 1
    runs: int = 5
    timeout_s: float | None = None


@dataclass
class SuiteConfig:
    """Configuration for a benchmark suite."""

    benchmarks: list[BenchmarkConfig] = field(default_factory=list)


def load_config(path: str | Path) -> SuiteConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated SuiteConfig object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML is invalid.
        ValueError: If the configuration is invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f) or {}

    return _parse_and_validate(raw_config)


def _parse_and_validate(raw: dict[str, Any]) -> SuiteConfig:
    """Parse raw config dict and validate it.

    Args:
        raw: Raw configuration dictionary from YAML.

    Returns:
        Validated SuiteConfig object.

    Raises:
        ValueError: If validation fails.
    """
    raw_benchmarks = raw.get("benchmarks")

    if not raw_benchmarks:
        raise ValueError("Configuration must contain a non-empty 'benchmarks' list.")

    if not isinstance(raw_benchmarks, list):
        raise ValueError("'benchmarks' must be a list.")

    benchmarks: list[BenchmarkConfig] = []
    seen_names: set[str] = set()

    for i, item in enumerate(raw_benchmarks):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark at index {i} must be a mapping.")

        benchmark = _parse_benchmark(item, index=i)

        if benchmark.name in seen_names:
            raise ValueError(f"Duplicate benchmark name: '{benchmark.name}'")
        seen_names.add(benchmark.name)

        benchmarks.append(benchmark)

    return SuiteConfig(benchmarks=benchmarks)


def _parse_benchmark(item: dict[str, Any], index: int) -> BenchmarkConfig:
    """Parse and validate a single benchmark configuration.

    Args:
        item: Raw benchmark dictionary.
        index: Index in the benchmarks list (for error messages).

    Returns:
        Validated BenchmarkConfig object.

    Raises:
        ValueError: If validation fails.
    """
    # Validate name
    name = item.get("name")
    if name is None:
        raise ValueError(f"Benchmark at index {index} is missing required field 'name'.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Benchmark at index {index}: 'name' must be a non-empty string.")
    name = name.strip()

    # Validate command
    command = item.get("command")
    if command is None:
        raise ValueError(f"Benchmark '{name}' is missing required field 'command'.")
    if not isinstance(command, list):
        raise ValueError(f"Benchmark '{name}': 'command' must be a list of strings.")
    if len(command) == 0:
        raise ValueError(f"Benchmark '{name}': 'command' must be a non-empty list.")
    if not all(isinstance(arg, str) for arg in command):
        raise ValueError(f"Benchmark '{name}': 'command' must contain only strings.")

    # Validate warmups
    warmups = item.get("warmups", 1)
    if not isinstance(warmups, int) or isinstance(warmups, bool):
        raise ValueError(f"Benchmark '{name}': 'warmups' must be an integer.")
    if warmups < 0:
        raise ValueError(f"Benchmark '{name}': 'warmups' must be >= 0.")

    # Validate runs
    runs = item.get("runs", 5)
    if not isinstance(runs, int) or isinstance(runs, bool):
        raise ValueError(f"Benchmark '{name}': 'runs' must be an integer.")
    if runs < 0:
        raise ValueError(f"Benchmark '{name}': 'runs' must be >= 0.")

    # Validate timeout_s
    timeout_s = item.get("timeout_s")
    if timeout_s is not None:
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            raise ValueError(f"Benchmark '{name}': 'timeout_s' must be a number or null.")
        if timeout_s <= 0:
            raise ValueError(f"Benchmark '{name}': 'timeout_s' must be positive.")

    return BenchmarkConfig(
        name=name,
        command=command,
        warmups=warmups,
        runs=runs,
        timeout_s=float(timeout_s) if timeout_s is not None else None,
    )
