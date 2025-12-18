"""Tests for profile_select.py - case selection helpers for profiling.

These tests verify filtering and selection logic without requiring perf.
"""

import json
import pytest
from pathlib import Path

from stoatix.profile_select import (
    filter_cases,
    load_cases_json,
    match_case_ids,
    select_top_regressions,
)


# Sample test data

SAMPLE_CASES = [
    {
        "bench_name": "sort_benchmark",
        "case_id": "abc123",
        "case_key": "n=1000",
        "params": {"n": 1000},
        "command": ["python", "sort.py", "1000"],
        "cwd": None,
        "env": {},
        "warmups": 1,
        "runs": 5,
        "retries": 0,
        "timeout_s": 30,
        "pin": {"cores": None},
    },
    {
        "bench_name": "sort_benchmark",
        "case_id": "def456",
        "case_key": "n=5000",
        "params": {"n": 5000},
        "command": ["python", "sort.py", "5000"],
        "cwd": None,
        "env": {},
        "warmups": 1,
        "runs": 5,
        "retries": 0,
        "timeout_s": 30,
        "pin": {"cores": None},
    },
    {
        "bench_name": "matrix_multiply",
        "case_id": "ghi789",
        "case_key": "size=100",
        "params": {"size": 100},
        "command": ["python", "matrix.py", "100"],
        "cwd": None,
        "env": {},
        "warmups": 2,
        "runs": 10,
        "retries": 1,
        "timeout_s": 60,
        "pin": {"cores": [0]},
    },
    {
        "bench_name": "matrix_multiply",
        "case_id": "jkl012",
        "case_key": "size=500",
        "params": {"size": 500},
        "command": ["python", "matrix.py", "500"],
        "cwd": None,
        "env": {},
        "warmups": 2,
        "runs": 10,
        "retries": 1,
        "timeout_s": 120,
        "pin": {"cores": [0]},
    },
]

SAMPLE_COMPARE = {
    "metadata": {
        "threshold": 0.05,
        "metric": "median_s",
    },
    "counts": {
        "regressed": 2,
        "improved": 1,
        "unchanged": 1,
        "added": 0,
        "removed": 0,
    },
    "rows": [
        {
            "bench_name": "sort_benchmark",
            "case_id": "abc123",
            "case_key": "n=1000",
            "pct_change": 0.15,  # 15% regression
            "classification": "regressed",
        },
        {
            "bench_name": "sort_benchmark",
            "case_id": "def456",
            "case_key": "n=5000",
            "pct_change": 0.08,  # 8% regression
            "classification": "regressed",
        },
        {
            "bench_name": "matrix_multiply",
            "case_id": "ghi789",
            "case_key": "size=100",
            "pct_change": -0.10,  # 10% improvement
            "classification": "improved",
        },
        {
            "bench_name": "matrix_multiply",
            "case_id": "jkl012",
            "case_key": "size=500",
            "pct_change": 0.02,  # 2% - unchanged
            "classification": "unchanged",
        },
    ],
}


class TestFilterCases:
    """Tests for filter_cases function."""

    def test_filter_by_bench_name(self):
        """Filter cases by benchmark name substring."""
        result = filter_cases(SAMPLE_CASES, bench="sort")
        assert len(result) == 2
        assert all(c["bench_name"] == "sort_benchmark" for c in result)

    def test_filter_by_bench_name_case_insensitive(self):
        """Filter is case-insensitive."""
        result = filter_cases(SAMPLE_CASES, bench="MATRIX")
        assert len(result) == 2
        assert all(c["bench_name"] == "matrix_multiply" for c in result)

    def test_filter_by_case_id_list(self):
        """Filter cases by case_id list."""
        result = filter_cases(SAMPLE_CASES, case_id=["abc123", "ghi789"])
        assert len(result) == 2
        case_ids = {c["case_id"] for c in result}
        assert case_ids == {"abc123", "ghi789"}

    def test_filter_by_case_key_contains(self):
        """Filter cases by case_key substring."""
        result = filter_cases(SAMPLE_CASES, case_key_contains="n=")
        assert len(result) == 2
        assert all("n=" in c["case_key"] for c in result)

    def test_filter_by_case_key_contains_case_insensitive(self):
        """case_key_contains is case-insensitive."""
        result = filter_cases(SAMPLE_CASES, case_key_contains="SIZE=")
        assert len(result) == 2
        assert all("size=" in c["case_key"] for c in result)

    def test_filter_combined_and_logic(self):
        """Multiple filters combine with AND logic."""
        result = filter_cases(
            SAMPLE_CASES,
            bench="matrix",
            case_key_contains="500",
        )
        assert len(result) == 1
        assert result[0]["case_id"] == "jkl012"

    def test_filter_no_match(self):
        """Filter returns empty list when no matches."""
        result = filter_cases(SAMPLE_CASES, bench="nonexistent")
        assert result == []

    def test_filter_preserves_order(self):
        """Filter preserves original case order."""
        result = filter_cases(SAMPLE_CASES, bench="sort")
        assert result[0]["case_id"] == "abc123"
        assert result[1]["case_id"] == "def456"


class TestSelectTopRegressions:
    """Tests for select_top_regressions function."""

    def test_select_top_regressions_default(self):
        """Select top regressions by pct_change descending."""
        result = select_top_regressions(SAMPLE_COMPARE, top=2)
        assert result == ["abc123", "def456"]  # 15% > 8%

    def test_select_top_regressions_limit(self):
        """Top parameter limits results."""
        result = select_top_regressions(SAMPLE_COMPARE, top=1)
        assert result == ["abc123"]

    def test_select_top_regressions_only_regressed(self):
        """Default only='regressed' filters to regressed cases."""
        result = select_top_regressions(SAMPLE_COMPARE, top=10, only="regressed")
        assert len(result) == 2
        assert "ghi789" not in result  # improved
        assert "jkl012" not in result  # unchanged

    def test_select_top_improved(self):
        """only='improved' returns improved cases sorted by pct_change asc."""
        result = select_top_regressions(SAMPLE_COMPARE, top=5, only="improved")
        assert result == ["ghi789"]

    def test_select_top_all(self):
        """only='all' returns all cases with pct_change, sorted by abs value."""
        result = select_top_regressions(SAMPLE_COMPARE, top=10, only="all")
        # Sorted by abs(pct_change): 0.15, 0.10, 0.08, 0.02
        assert result == ["abc123", "ghi789", "def456", "jkl012"]

    def test_select_empty_compare(self):
        """Empty compare returns empty list."""
        result = select_top_regressions({"rows": []}, top=5)
        assert result == []

    def test_select_no_regressions(self):
        """Returns empty when no regressions exist."""
        compare_no_reg = {
            "rows": [
                {"case_id": "x", "pct_change": -0.05, "classification": "improved"},
            ]
        }
        result = select_top_regressions(compare_no_reg, top=5, only="regressed")
        assert result == []


class TestMatchCaseIds:
    """Tests for match_case_ids function."""

    def test_match_preserves_order(self):
        """Matched cases are returned in case_ids order."""
        result = match_case_ids(SAMPLE_CASES, ["ghi789", "abc123"])
        assert len(result) == 2
        assert result[0]["case_id"] == "ghi789"
        assert result[1]["case_id"] == "abc123"

    def test_match_strict_raises_on_missing(self):
        """strict=True raises KeyError for missing case_ids."""
        with pytest.raises(KeyError) as exc_info:
            match_case_ids(SAMPLE_CASES, ["abc123", "nonexistent"], strict=True)
        assert "nonexistent" in str(exc_info.value)

    def test_match_non_strict_skips_missing(self):
        """strict=False silently skips missing case_ids."""
        result = match_case_ids(
            SAMPLE_CASES,
            ["abc123", "nonexistent", "def456"],
            strict=False,
        )
        assert len(result) == 2
        assert result[0]["case_id"] == "abc123"
        assert result[1]["case_id"] == "def456"

    def test_match_empty_case_ids(self):
        """Empty case_ids returns empty list."""
        result = match_case_ids(SAMPLE_CASES, [])
        assert result == []

    def test_match_all_missing_strict(self):
        """All missing case_ids raises with all IDs listed."""
        with pytest.raises(KeyError) as exc_info:
            match_case_ids(SAMPLE_CASES, ["x", "y", "z"], strict=True)
        assert "x" in str(exc_info.value)
        assert "y" in str(exc_info.value)
        assert "z" in str(exc_info.value)


class TestLoadCasesJson:
    """Tests for load_cases_json function."""

    def test_load_list_format(self, tmp_path: Path):
        """Load cases from JSON list format."""
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(json.dumps(SAMPLE_CASES), encoding="utf-8")

        result = load_cases_json(cases_file)
        assert len(result) == 4
        assert result[0]["case_id"] == "abc123"

    def test_load_object_with_cases_key(self, tmp_path: Path):
        """Load cases from JSON object with 'cases' key."""
        cases_file = tmp_path / "cases.json"
        data = {"suite_id": "test123", "cases": SAMPLE_CASES}
        cases_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_cases_json(cases_file)
        assert len(result) == 4
        assert result[0]["case_id"] == "abc123"

    def test_load_file_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_cases_json(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path: Path):
        """Raises JSONDecodeError for invalid JSON."""
        cases_file = tmp_path / "cases.json"
        cases_file.write_text("not valid json {{{", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_cases_json(cases_file)

    def test_load_object_missing_cases_key(self, tmp_path: Path):
        """Raises ValueError when object lacks 'cases' key."""
        cases_file = tmp_path / "cases.json"
        cases_file.write_text('{"other_key": []}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_cases_json(cases_file)
        assert "cases" in str(exc_info.value)

    def test_load_invalid_structure(self, tmp_path: Path):
        """Raises ValueError for unexpected structure."""
        cases_file = tmp_path / "cases.json"
        cases_file.write_text('"just a string"', encoding="utf-8")

        with pytest.raises(ValueError):
            load_cases_json(cases_file)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_select_and_match_workflow(self):
        """Typical workflow: select top regressions then match to cases."""
        # Select top 2 regressions
        regression_ids = select_top_regressions(SAMPLE_COMPARE, top=2)
        assert regression_ids == ["abc123", "def456"]

        # Match to cases
        matched = match_case_ids(SAMPLE_CASES, regression_ids)
        assert len(matched) == 2
        assert matched[0]["case_id"] == "abc123"
        assert matched[1]["case_id"] == "def456"

    def test_filter_then_match(self):
        """Filter by bench, then match specific IDs."""
        # Filter to matrix benchmarks
        matrix_cases = filter_cases(SAMPLE_CASES, bench="matrix")
        assert len(matrix_cases) == 2

        # Match specific ID from filtered set
        matched = match_case_ids(matrix_cases, ["jkl012"])
        assert len(matched) == 1
        assert matched[0]["bench_name"] == "matrix_multiply"
