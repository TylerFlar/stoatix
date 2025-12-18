"""Templating and matrix expansion for benchmark cases."""

import hashlib
import itertools
import json
import string
from dataclasses import dataclass
from typing import Any

from stoatix.config import BenchmarkConfig, PinConfig, SuiteConfig


# Type alias for scalar values in matrix
Scalar = str | int | float | bool


@dataclass
class CaseSpec:
    """Specification for a single benchmark case (after matrix expansion)."""

    bench_name: str
    params: dict[str, Scalar]
    command: list[str]
    cwd: str | None
    env: dict[str, str]
    warmups: int
    runs: int
    timeout_s: float | None
    pin: PinConfig
    case_key: str
    case_id: str


class SafeFormatMapping(dict[str, Any]):
    """A mapping that raises ValueError for missing keys during str.format_map().

    This ensures that only explicitly provided parameters can be substituted,
    and any missing placeholder raises a clear error.
    """

    def __init__(self, params: dict[str, Any], context: str = "") -> None:
        super().__init__(params)
        self._context = context

    def __missing__(self, key: str) -> str:
        raise ValueError(
            f"Template placeholder '{{{key}}}' not found in params. "
            f"Available: {list(self.keys())}."
        )


def render_template(s: str, params: dict[str, Any]) -> str:
    """Render a template string with safe parameter substitution.

    Uses {param} style placeholders. Only simple field names are supported;
    attribute access ({obj.attr}) and indexing ({arr[0]}) are not allowed.

    Args:
        s: Template string with {placeholder} markers.
        params: Dictionary of parameter values to substitute.

    Returns:
        Rendered string with placeholders replaced.

    Raises:
        ValueError: If a placeholder is not found in params or uses
                    unsupported syntax (attribute access, indexing).
    """
    # Check for unsupported placeholder syntax before formatting
    formatter = string.Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(s):
        if field_name is None:
            continue
        # field_name can contain dots (attr access) or brackets (indexing)
        # We only allow simple names
        if "." in field_name or "[" in field_name:
            raise ValueError(
                f"Template placeholder '{{{field_name}}}' uses unsupported syntax. "
                f"Only simple placeholders like '{{name}}' are allowed."
            )
        if field_name and field_name not in params:
            raise ValueError(
                f"Template placeholder '{{{field_name}}}' not found in params. "
                f"Available: {list(params.keys())}."
            )

    # Convert all param values to strings for format_map
    str_params = {k: str(v) for k, v in params.items()}
    return s.format_map(SafeFormatMapping(str_params))


def _render_command(command: list[str], params: dict[str, Any]) -> list[str]:
    """Render all elements in a command list."""
    return [render_template(arg, params) for arg in command]


def _render_env(env: dict[str, str], params: dict[str, Any]) -> dict[str, str]:
    """Render all values in an env dictionary."""
    return {k: render_template(v, params) for k, v in env.items()}


def _render_cwd(cwd: str | None, params: dict[str, Any]) -> str | None:
    """Render cwd if present."""
    if cwd is None:
        return None
    return render_template(cwd, params)


def _compute_case_key(params: dict[str, Scalar]) -> str:
    """Compute a deterministic case key from parameters.

    Format: "k1=v1,k2=v2" with keys sorted alphabetically.
    Values are converted to strings consistently.

    Args:
        params: Parameter dictionary.

    Returns:
        Case key string, or empty string if no params.
    """
    if not params:
        return ""
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return ",".join(parts)


def _compute_case_id(bench_name: str, params: dict[str, Scalar]) -> str:
    """Compute a stable short hash ID for a case.

    Uses SHA-1 of canonical JSON representation.

    Args:
        bench_name: Name of the benchmark.
        params: Parameter dictionary.

    Returns:
        First 12 hex characters of SHA-1 hash.
    """
    # Create canonical JSON with sorted keys for stability
    canonical = json.dumps(
        {"bench": bench_name, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    hash_bytes = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    return hash_bytes[:12]


def expand_benchmark_cases(bench: BenchmarkConfig) -> list[CaseSpec]:
    """Expand a benchmark configuration into individual cases.

    If matrix is empty or missing, produces a single case with empty params.
    Otherwise, produces cartesian product of all matrix dimensions.

    Expansion order:
    - Matrix keys are iterated in sorted order
    - Each key's values are iterated in YAML order (as given)

    Args:
        bench: Benchmark configuration to expand.

    Returns:
        List of CaseSpec objects, one per parameter combination.
    """
    matrix = bench.matrix

    if not matrix:
        # No matrix: single case with empty params
        params: dict[str, Scalar] = {}
        case_key = _compute_case_key(params)
        case_id = _compute_case_id(bench.name, params)

        return [
            CaseSpec(
                bench_name=bench.name,
                params=params,
                command=_render_command(bench.command, params),
                cwd=_render_cwd(bench.cwd, params),
                env=_render_env(bench.env, params),
                warmups=bench.warmups,
                runs=bench.runs,
                timeout_s=bench.timeout_s,
                pin=bench.pin,
                case_key=case_key,
                case_id=case_id,
            )
        ]

    # Get keys in sorted order for deterministic iteration
    sorted_keys = sorted(matrix.keys())

    # Get value lists in the same order
    value_lists = [matrix[k] for k in sorted_keys]

    # Cartesian product
    cases: list[CaseSpec] = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(sorted_keys, combo))
        case_key = _compute_case_key(params)
        case_id = _compute_case_id(bench.name, params)

        cases.append(
            CaseSpec(
                bench_name=bench.name,
                params=params,
                command=_render_command(bench.command, params),
                cwd=_render_cwd(bench.cwd, params),
                env=_render_env(bench.env, params),
                warmups=bench.warmups,
                runs=bench.runs,
                timeout_s=bench.timeout_s,
                pin=bench.pin,
                case_key=case_key,
                case_id=case_id,
            )
        )

    return cases


def expand_suite(config: SuiteConfig) -> list[CaseSpec]:
    """Expand all benchmarks in a suite into individual cases.

    Preserves benchmark order as written in YAML.
    Within each benchmark, preserves deterministic case order from expansion.

    Args:
        config: Suite configuration to expand.

    Returns:
        List of all CaseSpec objects from all benchmarks.
    """
    all_cases: list[CaseSpec] = []

    for bench in config.benchmarks:
        cases = expand_benchmark_cases(bench)
        all_cases.extend(cases)

    return all_cases
