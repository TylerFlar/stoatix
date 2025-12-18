"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class PinConfig:
    """CPU pinning configuration."""

    strategy: Literal["none", "taskset"] = "none"
    cores: list[int] = field(default_factory=list)


@dataclass
class DefaultsConfig:
    """Default configuration values for benchmarks."""

    warmups: int = 1
    runs: int = 5
    retries: int = 0
    timeout_s: float | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    pin: PinConfig = field(default_factory=PinConfig)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    command: list[str]
    warmups: int = 1
    runs: int = 5
    retries: int = 0
    timeout_s: float | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    pin: PinConfig = field(default_factory=PinConfig)
    matrix: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class SuiteConfig:
    """Configuration for a benchmark suite."""

    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    benchmarks: list[BenchmarkConfig] = field(default_factory=list)

    def to_resolved_dict(self) -> dict[str, Any]:
        """Produce a resolved config dict with all defaults explicit.

        Returns:
            Dictionary suitable for writing to YAML, with:
            - defaults section fully filled with explicit values
            - each benchmark showing inherited + overridden values
        """
        # Build defaults dict with all explicit values
        defaults_dict: dict[str, Any] = {
            "warmups": self.defaults.warmups,
            "runs": self.defaults.runs,
            "retries": self.defaults.retries,
            "timeout_s": self.defaults.timeout_s,
            "cwd": self.defaults.cwd,
            "env": dict(self.defaults.env) if self.defaults.env else {},
            "pin": {
                "strategy": self.defaults.pin.strategy,
                "cores": list(self.defaults.pin.cores) if self.defaults.pin.cores else [],
            },
        }

        # Build benchmarks list with resolved values
        benchmarks_list: list[dict[str, Any]] = []
        for bench in self.benchmarks:
            bench_dict: dict[str, Any] = {
                "name": bench.name,
                "command": list(bench.command),
                "warmups": bench.warmups,
                "runs": bench.runs,
                "retries": bench.retries,
                "timeout_s": bench.timeout_s,
                "cwd": bench.cwd,
                "env": dict(bench.env) if bench.env else {},
                "pin": {
                    "strategy": bench.pin.strategy,
                    "cores": list(bench.pin.cores) if bench.pin.cores else [],
                },
            }
            if bench.matrix:
                bench_dict["matrix"] = {k: list(v) for k, v in bench.matrix.items()}
            benchmarks_list.append(bench_dict)

        return {
            "defaults": defaults_dict,
            "benchmarks": benchmarks_list,
        }


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
    # Validate top-level keys
    allowed_top_level = {"defaults", "benchmarks"}
    for key in raw:
        if key not in allowed_top_level:
            raise ValueError(
                f"Unknown top-level key '{key}'. Allowed keys: {sorted(allowed_top_level)}."
            )

    # Parse defaults
    defaults = _parse_defaults(raw.get("defaults", {}))

    # Parse benchmarks
    raw_benchmarks = raw.get("benchmarks")

    if not raw_benchmarks:
        raise ValueError("Configuration must contain a non-empty 'benchmarks' list.")

    if not isinstance(raw_benchmarks, list):
        raise ValueError("'benchmarks' must be a list.")

    benchmarks: list[BenchmarkConfig] = []
    seen_names: set[str] = set()

    for i, item in enumerate(raw_benchmarks):
        if not isinstance(item, dict):
            raise ValueError(f"benchmarks[{i}]: must be a mapping.")

        benchmark = _parse_benchmark(item, index=i, defaults=defaults)

        if benchmark.name in seen_names:
            raise ValueError(f"benchmarks[{i}]: duplicate name '{benchmark.name}'.")
        seen_names.add(benchmark.name)

        benchmarks.append(benchmark)

    return SuiteConfig(defaults=defaults, benchmarks=benchmarks)


def _parse_defaults(raw: Any) -> DefaultsConfig:
    """Parse and validate the defaults section.

    Args:
        raw: Raw defaults dictionary from YAML.

    Returns:
        Validated DefaultsConfig object.

    Raises:
        ValueError: If validation fails.
    """
    if raw is None:
        return DefaultsConfig()

    if not isinstance(raw, dict):
        raise ValueError("'defaults' must be a mapping.")

    # Validate defaults keys
    allowed_defaults = {"warmups", "runs", "retries", "timeout_s", "cwd", "env", "pin"}
    for key in raw:
        if key not in allowed_defaults:
            raise ValueError(
                f"defaults: unknown field '{key}'. Allowed fields: {sorted(allowed_defaults)}."
            )

    warmups = _parse_int_field(raw, "warmups", "defaults", default=1, min_val=0)
    runs = _parse_int_field(raw, "runs", "defaults", default=5, min_val=0)
    retries = _parse_int_field(raw, "retries", "defaults", default=0, min_val=0)
    timeout_s = _parse_timeout(raw, "defaults")
    cwd = _parse_optional_str(raw, "cwd", "defaults")
    env = _parse_env(raw, "defaults")
    pin = _parse_pin(raw.get("pin", {}), "defaults")

    return DefaultsConfig(
        warmups=warmups,
        runs=runs,
        retries=retries,
        timeout_s=timeout_s,
        cwd=cwd,
        env=env,
        pin=pin,
    )


def _parse_pin(raw: Any, context: str) -> PinConfig:
    """Parse and validate pin configuration.

    Args:
        raw: Raw pin dictionary from YAML.
        context: Context string for error messages (e.g., "defaults" or "benchmarks[0]").

    Returns:
        Validated PinConfig object.

    Raises:
        ValueError: If validation fails.
    """
    if raw is None:
        return PinConfig()

    if not isinstance(raw, dict):
        raise ValueError(f"{context}.pin: must be a mapping.")

    strategy = raw.get("strategy", "none")
    if strategy not in ("none", "taskset"):
        raise ValueError(
            f"{context}.pin.strategy: must be 'none' or 'taskset', got '{strategy}'."
        )

    cores: list[int] = []
    if strategy == "taskset":
        raw_cores = raw.get("cores")
        if raw_cores is None:
            raise ValueError(
                f"{context}.pin.cores: required when strategy is 'taskset'."
            )
        if not isinstance(raw_cores, list):
            raise ValueError(f"{context}.pin.cores: must be a list of integers.")
        if len(raw_cores) == 0:
            raise ValueError(f"{context}.pin.cores: must be a non-empty list.")
        for j, core in enumerate(raw_cores):
            if not isinstance(core, int) or isinstance(core, bool):
                raise ValueError(
                    f"{context}.pin.cores[{j}]: must be an integer, got {type(core).__name__}."
                )
            if core < 0:
                raise ValueError(f"{context}.pin.cores[{j}]: must be >= 0, got {core}.")
        cores = raw_cores
    elif "cores" in raw:
        # cores specified but strategy is "none" - that's fine, just ignore or warn
        pass

    return PinConfig(strategy=strategy, cores=cores)


def _parse_benchmark(
    item: dict[str, Any], index: int, defaults: DefaultsConfig
) -> BenchmarkConfig:
    """Parse and validate a single benchmark configuration.

    Args:
        item: Raw benchmark dictionary.
        index: Index in the benchmarks list (for error messages).
        defaults: Default values to use for missing fields.

    Returns:
        Validated BenchmarkConfig object.

    Raises:
        ValueError: If validation fails.
    """
    context = f"benchmarks[{index}]"

    # Get name first for better error messages
    name = item.get("name")
    name_str = f"'{name}'" if name else "(unnamed)"

    # Validate benchmark keys
    allowed_benchmark = {
        "name", "command", "warmups", "runs", "retries", "timeout_s", "cwd", "env", "pin", "matrix"
    }
    for key in item:
        if key not in allowed_benchmark:
            raise ValueError(
                f"{context} {name_str}: unknown field '{key}'. "
                f"Allowed fields: {sorted(allowed_benchmark)}."
            )

    # Validate name
    if name is None:
        raise ValueError(f"{context}: missing required field 'name'.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{context}.name: must be a non-empty string.")
    name = name.strip()

    # Validate command
    command = item.get("command")
    if command is None:
        raise ValueError(f"{context} ('{name}'): missing required field 'command'.")
    if not isinstance(command, list):
        raise ValueError(f"{context}.command: must be a list of strings.")
    if len(command) == 0:
        raise ValueError(f"{context}.command: must be a non-empty list.")
    for j, arg in enumerate(command):
        if not isinstance(arg, str):
            raise ValueError(
                f"{context}.command[{j}]: must be a string, got {type(arg).__name__}."
            )

    # Parse with defaults
    warmups = _parse_int_field(
        item, "warmups", context, default=defaults.warmups, min_val=0
    )
    runs = _parse_int_field(item, "runs", context, default=defaults.runs, min_val=0)
    retries = _parse_int_field(
        item, "retries", context, default=defaults.retries, min_val=0
    )
    timeout_s = _parse_timeout(item, context, default=defaults.timeout_s)
    cwd = _parse_optional_str(item, "cwd", context, default=defaults.cwd)

    # Merge env: defaults + benchmark-specific
    env = dict(defaults.env)
    env.update(_parse_env(item, context))

    # Parse pin with defaults
    if "pin" in item:
        pin = _parse_pin(item["pin"], context)
    else:
        pin = defaults.pin

    # Parse matrix
    matrix = _parse_matrix(item, context)

    return BenchmarkConfig(
        name=name,
        command=command,
        warmups=warmups,
        runs=runs,
        retries=retries,
        timeout_s=timeout_s,
        cwd=cwd,
        env=env,
        pin=pin,
        matrix=matrix,
    )


def _parse_int_field(
    raw: dict[str, Any],
    field_name: str,
    context: str,
    default: int,
    min_val: int | None = None,
) -> int:
    """Parse and validate an integer field.

    Args:
        raw: Raw dictionary containing the field.
        field_name: Name of the field.
        context: Context string for error messages.
        default: Default value if field is not present.
        min_val: Minimum allowed value (inclusive).

    Returns:
        Validated integer value.

    Raises:
        ValueError: If validation fails.
    """
    value = raw.get(field_name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{context}.{field_name}: must be an integer, got {type(value).__name__}."
        )
    if min_val is not None and value < min_val:
        raise ValueError(f"{context}.{field_name}: must be >= {min_val}, got {value}.")
    return value


def _parse_timeout(
    raw: dict[str, Any], context: str, default: float | None = None
) -> float | None:
    """Parse and validate a timeout field.

    Args:
        raw: Raw dictionary containing the field.
        context: Context string for error messages.
        default: Default value if field is not present.

    Returns:
        Validated timeout value or None.

    Raises:
        ValueError: If validation fails.
    """
    value = raw.get("timeout_s", default)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"{context}.timeout_s: must be a number or null, got {type(value).__name__}."
        )
    if value <= 0:
        raise ValueError(f"{context}.timeout_s: must be positive, got {value}.")
    return float(value)


def _parse_optional_str(
    raw: dict[str, Any], field_name: str, context: str, default: str | None = None
) -> str | None:
    """Parse and validate an optional string field.

    Args:
        raw: Raw dictionary containing the field.
        field_name: Name of the field.
        context: Context string for error messages.
        default: Default value if field is not present.

    Returns:
        String value or None.

    Raises:
        ValueError: If validation fails.
    """
    value = raw.get(field_name, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{context}.{field_name}: must be a string or null, got {type(value).__name__}."
        )
    return value


def _parse_env(raw: dict[str, Any], context: str) -> dict[str, str]:
    """Parse and validate an env dictionary.

    Args:
        raw: Raw dictionary containing the env field.
        context: Context string for error messages.

    Returns:
        Validated environment dictionary.

    Raises:
        ValueError: If validation fails.
    """
    env = raw.get("env", {})
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise ValueError(f"{context}.env: must be a mapping.")
    result: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{context}.env: keys must be strings, got {type(key).__name__}."
            )
        if not isinstance(value, str):
            # Allow numbers to be converted to strings
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            else:
                raise ValueError(
                    f"{context}.env.{key}: value must be a string, got {type(value).__name__}."
                )
        result[key] = value
    return result


def _parse_matrix(raw: dict[str, Any], context: str) -> dict[str, list[Any]]:
    """Parse and validate a matrix dictionary.

    Args:
        raw: Raw dictionary containing the matrix field.
        context: Context string for error messages.

    Returns:
        Validated matrix dictionary.

    Raises:
        ValueError: If validation fails.
    """
    matrix = raw.get("matrix", {})
    if matrix is None:
        return {}
    if not isinstance(matrix, dict):
        raise ValueError(f"{context}.matrix: must be a mapping.")

    result: dict[str, list[Any]] = {}
    for key, values in matrix.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{context}.matrix: keys must be strings, got {type(key).__name__}."
            )
        if not isinstance(values, list):
            raise ValueError(
                f"{context}.matrix.{key}: must be a list of values."
            )
        if len(values) == 0:
            raise ValueError(
                f"{context}.matrix.{key}: must be a non-empty list."
            )
        # Validate that values are scalars (str, int, float, bool)
        for i, val in enumerate(values):
            if not isinstance(val, (str, int, float, bool)) or val is None:
                raise ValueError(
                    f"{context}.matrix.{key}[{i}]: must be a scalar (str, int, float, bool)."
                )
        result[key] = values
    return result
