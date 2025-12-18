"""Case selection helpers for deep profiling.

Provides utilities to filter and select benchmark cases for detailed profiling
based on benchmark names, case identifiers, or regression analysis results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from stoatix.plan import CaseSpec

__all__ = [
    "filter_cases",
    "select_top_regressions",
    "match_case_ids",
    "load_cases_json",
]


def filter_cases(
    cases: list[CaseSpec | dict[str, Any]],
    *,
    bench: str | None = None,
    case_id: list[str] | None = None,
    case_key_contains: str | None = None,
) -> list[CaseSpec | dict[str, Any]]:
    """Filter cases by benchmark name, case ID, or case key substring.

    Multiple filters are combined with AND logic.

    Args:
        cases: List of CaseSpec objects or dicts with case data.
        bench: Filter to cases where bench_name contains this substring
            (case-insensitive).
        case_id: Filter to cases whose case_id is in this list.
        case_key_contains: Filter to cases where case_key contains this
            substring (case-insensitive).

    Returns:
        Filtered list of cases matching all provided criteria.
        Original order is preserved.

    Examples:
        >>> filter_cases(cases, bench="sort")
        >>> filter_cases(cases, case_id=["abc123", "def456"])
        >>> filter_cases(cases, bench="matrix", case_key_contains="n=1000")
    """
    result = list(cases)

    if bench is not None:
        bench_lower = bench.lower()
        result = [
            c
            for c in result
            if bench_lower in _get_attr(c, "bench_name", "").lower()
        ]

    if case_id is not None:
        case_id_set = set(case_id)
        result = [c for c in result if _get_attr(c, "case_id", "") in case_id_set]

    if case_key_contains is not None:
        key_lower = case_key_contains.lower()
        result = [
            c for c in result if key_lower in _get_attr(c, "case_key", "").lower()
        ]

    return result


def select_top_regressions(
    compare_result: dict[str, Any],
    *,
    top: int = 5,
    only: Literal["regressed", "improved", "all"] = "regressed",
) -> list[str]:
    """Select case IDs of top regressions (or improvements) from compare output.

    Args:
        compare_result: Output from compare_runs() containing 'rows' list.
        top: Maximum number of case IDs to return.
        only: Which classifications to consider:
            - "regressed": Only regressed cases (positive pct_change).
            - "improved": Only improved cases (negative pct_change).
            - "all": All cases with non-null pct_change.

    Returns:
        List of case_ids sorted by absolute pct_change descending,
        limited to 'top' entries. For "regressed", sorts by pct_change desc.
        For "improved", sorts by pct_change asc (most negative first).
        For "all", sorts by absolute pct_change desc.

    Examples:
        >>> compare = compare_runs("main.jsonl", "pr.jsonl")
        >>> case_ids = select_top_regressions(compare, top=3)
        >>> case_ids = select_top_regressions(compare, top=5, only="improved")
    """
    rows = compare_result.get("rows", [])

    # Filter by classification
    if only == "regressed":
        filtered = [r for r in rows if r.get("classification") == "regressed"]
    elif only == "improved":
        filtered = [r for r in rows if r.get("classification") == "improved"]
    else:  # all
        filtered = [r for r in rows if r.get("pct_change") is not None]

    # Sort by pct_change
    if only == "regressed":
        # Highest positive first
        sorted_rows = sorted(
            filtered, key=lambda r: r.get("pct_change", 0), reverse=True
        )
    elif only == "improved":
        # Most negative first (largest improvement)
        sorted_rows = sorted(filtered, key=lambda r: r.get("pct_change", 0))
    else:
        # Absolute value, descending
        sorted_rows = sorted(
            filtered, key=lambda r: abs(r.get("pct_change", 0) or 0), reverse=True
        )

    # Extract case_ids
    return [r["case_id"] for r in sorted_rows[:top] if "case_id" in r]


def match_case_ids(
    cases: list[CaseSpec | dict[str, Any]],
    case_ids: list[str],
    *,
    strict: bool = True,
) -> list[CaseSpec | dict[str, Any]]:
    """Match cases by case_id, preserving order of case_ids.

    Args:
        cases: List of CaseSpec objects or dicts with case data.
        case_ids: List of case_ids to match, in desired output order.
        strict: If True, raise KeyError if any case_id is not found.
            If False, silently skip missing case_ids.

    Returns:
        List of cases matching the provided case_ids, in the same order
        as the case_ids list.

    Raises:
        KeyError: If strict=True and any case_id is not found in cases.

    Examples:
        >>> cases = load_cases_json("cases.json")
        >>> selected = match_case_ids(cases, ["abc123", "def456"])
    """
    # Build index: case_id -> case
    index: dict[str, CaseSpec | dict[str, Any]] = {}
    for c in cases:
        cid = _get_attr(c, "case_id", "")
        if cid:
            index[cid] = c

    result: list[CaseSpec | dict[str, Any]] = []
    missing: list[str] = []

    for cid in case_ids:
        if cid in index:
            result.append(index[cid])
        elif strict:
            missing.append(cid)

    if missing:
        raise KeyError(f"Case IDs not found: {missing}")

    return result


def load_cases_json(path: Path | str) -> list[dict[str, Any]]:
    """Load cases from a JSON file.

    The JSON file should contain either:
    - A list of case objects directly
    - An object with a 'cases' key containing the list

    Args:
        path: Path to the cases JSON file.

    Returns:
        List of case dicts with at minimum 'bench_name', 'case_id',
        'case_key', and 'command' fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the JSON structure is invalid.

    Examples:
        >>> cases = load_cases_json("cases.json")
        >>> for case in cases:
        ...     print(case["bench_name"], case["case_id"])
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and object with 'cases' key
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        if "cases" in data:
            cases = data["cases"]
        else:
            raise ValueError(
                "JSON object must have a 'cases' key containing the case list"
            )
    else:
        raise ValueError(f"Expected list or object, got {type(data).__name__}")

    if not isinstance(cases, list):
        raise ValueError(f"Cases must be a list, got {type(cases).__name__}")

    return cases


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
