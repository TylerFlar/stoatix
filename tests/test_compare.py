"""Tests for compare module."""

import json
from pathlib import Path

import pytest

from stoatix.compare import (
    compare_runs,
    render_compare_markdown,
    write_compare_json,
    write_compare_md,
)


def _make_record(
    bench_name: str,
    case_id: str,
    case_key: str,
    params: dict,
    iteration: int,
    elapsed_s: float,
    ok: bool = True,
    run_kind: str = "measured",
) -> dict:
    """Create a minimal result record for testing."""
    return {
        "suite_id": "test-suite",
        "bench_name": bench_name,
        "case_id": case_id,
        "case_key": case_key,
        "params": params,
        "command": ["echo", "test"],
        "cwd": None,
        "env_overrides": {},
        "run_kind": run_kind,
        "iteration": iteration,
        "attempt": 1,
        "started_at_utc": "2025-01-01T00:00:00+00:00",
        "ended_at_utc": "2025-01-01T00:00:01+00:00",
        "elapsed_s": elapsed_s,
        "exit_code": 0 if ok else 1,
        "timed_out": False,
        "ok": ok,
        "stdout_len": 0,
        "stderr_len": 0,
        "stdout_trunc": "",
        "stderr_trunc": "",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_identical_runs_all_unchanged(self, tmp_path: Path) -> None:
        """When runs are identical, all cases are classified as unchanged."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            threshold=0.05,
        )

        assert result["counts"]["unchanged"] == 1
        assert result["counts"]["regressed"] == 0
        assert result["counts"]["improved"] == 0
        assert len(result["rows"]) == 1
        assert result["rows"][0]["classification"] == "unchanged"

    def test_regression_detected(self, tmp_path: Path) -> None:
        """When PR is slower by > threshold, case is classified as regressed."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.120),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.120),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.120),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            threshold=0.05,  # 5% threshold, 20% regression
        )

        assert result["counts"]["regressed"] == 1
        assert result["rows"][0]["classification"] == "regressed"
        assert result["rows"][0]["pct_change"] == pytest.approx(0.20, rel=0.01)

    def test_improvement_detected(self, tmp_path: Path) -> None:
        """When PR is faster by > threshold, case is classified as improved."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.080),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.080),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.080),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            threshold=0.05,  # 5% threshold, 20% improvement
        )

        assert result["counts"]["improved"] == 1
        assert result["rows"][0]["classification"] == "improved"
        assert result["rows"][0]["pct_change"] == pytest.approx(-0.20, rel=0.01)

    def test_added_case_detected(self, tmp_path: Path) -> None:
        """When case only exists in PR, it's classified as added."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
            # New case only in PR
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 1, 0.050),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 2, 0.050),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 3, 0.050),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        result = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        assert result["counts"]["added"] == 1
        assert result["counts"]["unchanged"] == 1

        added_row = next(r for r in result["rows"] if r["classification"] == "added")
        assert added_row["bench_name"] == "bench-b"
        assert added_row["pct_change"] is None

    def test_removed_case_detected(self, tmp_path: Path) -> None:
        """When case only exists in main, it's classified as removed."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
            # Case only in main
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 1, 0.050),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 2, 0.050),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 3, 0.050),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        result = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        assert result["counts"]["removed"] == 1
        assert result["counts"]["unchanged"] == 1

        removed_row = next(r for r in result["rows"] if r["classification"] == "removed")
        assert removed_row["bench_name"] == "bench-b"
        assert removed_row["pct_change"] is None


class TestNeedsAttention:
    """Tests for needs_attention flagging."""

    def test_high_cv_triggers_needs_attention(self, tmp_path: Path) -> None:
        """When CV (stdev/median) is high, case needs attention."""
        # High variance in main
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.050),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.150),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            noise_cv_threshold=0.05,  # Low threshold to trigger
        )

        assert result["rows"][0]["needs_attention"] is True
        assert result["counts"]["needs_attention"] == 1

    def test_low_n_ok_triggers_needs_attention(self, tmp_path: Path) -> None:
        """When n_ok is below min_ok, case needs attention."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            min_ok=3,  # main only has 2
        )

        assert result["rows"][0]["needs_attention"] is True
        assert result["rows"][0]["main_n_ok"] == 2
        assert result["rows"][0]["pr_n_ok"] == 3

    def test_stable_results_no_attention_needed(self, tmp_path: Path) -> None:
        """When results are stable, no attention flag is set."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 4, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 5, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            noise_cv_threshold=0.05,
            noise_p95_ratio_threshold=1.10,
            min_ok=3,
        )

        assert result["rows"][0]["needs_attention"] is False
        assert result["counts"]["needs_attention"] == 0


class TestMetadataAndOutput:
    """Tests for metadata and output structure."""

    def test_metadata_contains_parameters(self, tmp_path: Path) -> None:
        """Result metadata contains all comparison parameters."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result = compare_runs(
            tmp_path / "main.jsonl",
            tmp_path / "pr.jsonl",
            threshold=0.10,
            outliers="none",
            metric="mean_s",
            noise_cv_threshold=0.08,
            noise_p95_ratio_threshold=1.15,
            min_ok=5,
        )

        meta = result["metadata"]
        assert meta["threshold"] == 0.10
        assert meta["outliers"] == "none"
        assert meta["metric"] == "mean_s"
        assert meta["noise_cv_threshold"] == 0.08
        assert meta["noise_p95_ratio_threshold"] == 1.15
        assert meta["min_ok"] == 5
        assert "created_at_utc" in meta
        assert str(tmp_path / "main.jsonl") in meta["main_path"]
        assert str(tmp_path / "pr.jsonl") in meta["pr_path"]

    def test_rows_sorted_deterministically(self, tmp_path: Path) -> None:
        """Rows are sorted by (bench_name, case_key, case_id)."""
        records = [
            _make_record("bench-z", "case-1", "n=1", {"n": 1}, 1, 0.100),
            _make_record("bench-z", "case-1", "n=1", {"n": 1}, 2, 0.100),
            _make_record("bench-z", "case-1", "n=1", {"n": 1}, 3, 0.100),
            _make_record("bench-a", "case-2", "n=2", {"n": 2}, 1, 0.100),
            _make_record("bench-a", "case-2", "n=2", {"n": 2}, 2, 0.100),
            _make_record("bench-a", "case-2", "n=2", {"n": 2}, 3, 0.100),
            _make_record("bench-a", "case-1", "n=1", {"n": 1}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=1", {"n": 1}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=1", {"n": 1}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        # Should be sorted: bench-a/n=1, bench-a/n=2, bench-z/n=1
        assert result["rows"][0]["bench_name"] == "bench-a"
        assert result["rows"][0]["case_key"] == "n=1"
        assert result["rows"][1]["bench_name"] == "bench-a"
        assert result["rows"][1]["case_key"] == "n=2"
        assert result["rows"][2]["bench_name"] == "bench-z"

    def test_params_json_deterministic(self, tmp_path: Path) -> None:
        """params_json is deterministic with sorted keys."""
        records = [
            _make_record("bench-a", "case-1", "b=2,a=1", {"b": 2, "a": 1}, 1, 0.100),
            _make_record("bench-a", "case-1", "b=2,a=1", {"b": 2, "a": 1}, 2, 0.100),
            _make_record("bench-a", "case-1", "b=2,a=1", {"b": 2, "a": 1}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        # Keys should be sorted in JSON
        assert result["rows"][0]["params_json"] == '{"a": 1, "b": 2}'

    def test_row_contains_all_expected_fields(self, tmp_path: Path) -> None:
        """Each row contains all expected fields."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.105),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.095),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        row = result["rows"][0]

        # Required fields
        assert "bench_name" in row
        assert "case_id" in row
        assert "case_key" in row
        assert "params_json" in row
        assert "main_median_s" in row
        assert "pr_median_s" in row
        assert "pct_change" in row
        assert "classification" in row
        assert "needs_attention" in row
        assert "main_n_ok" in row
        assert "pr_n_ok" in row
        assert "main_stdev_s" in row
        assert "pr_stdev_s" in row
        assert "main_p95_s" in row
        assert "pr_p95_s" in row


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_main_file_raises(self, tmp_path: Path) -> None:
        """Missing main results file raises FileNotFoundError."""
        _write_jsonl(tmp_path / "pr.jsonl", [])

        with pytest.raises(FileNotFoundError, match="Main results"):
            compare_runs(tmp_path / "missing.jsonl", tmp_path / "pr.jsonl")

    def test_missing_pr_file_raises(self, tmp_path: Path) -> None:
        """Missing PR results file raises FileNotFoundError."""
        _write_jsonl(tmp_path / "main.jsonl", [])

        with pytest.raises(FileNotFoundError, match="PR results"):
            compare_runs(tmp_path / "main.jsonl", tmp_path / "missing.jsonl")


class TestDeterminism:
    """Tests for deterministic output."""

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        """Running compare_runs twice produces identical output (except timestamp)."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.105),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.095),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 1, 0.050),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 2, 0.052),
            _make_record("bench-b", "case-2", "x=1", {"x": 1}, 3, 0.048),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        result1 = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        result2 = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        # Compare rows (excluding timestamp which will differ)
        assert result1["rows"] == result2["rows"]
        assert result1["counts"] == result2["counts"]

        # Metadata should match except created_at_utc
        meta1 = {k: v for k, v in result1["metadata"].items() if k != "created_at_utc"}
        meta2 = {k: v for k, v in result2["metadata"].items() if k != "created_at_utc"}
        assert meta1 == meta2


class TestRenderCompareMarkdown:
    """Tests for render_compare_markdown function."""

    def test_renders_header_with_metadata(self, tmp_path: Path) -> None:
        """Markdown includes header with metric, threshold, and outliers."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        md = render_compare_markdown(compare)

        assert "## Benchmark Comparison" in md
        assert "median_s" in md
        assert "5.0%" in md  # threshold
        assert "iqr" in md

    def test_renders_summary_counts(self, tmp_path: Path) -> None:
        """Markdown includes summary counts with emojis."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.120),  # regressed
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.120),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.120),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        md = render_compare_markdown(compare)

        assert "🔴" in md
        assert "regressed" in md

    def test_renders_table_with_correct_columns(self, tmp_path: Path) -> None:
        """Markdown includes table with expected columns."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        md = render_compare_markdown(compare)

        assert "| bench_name |" in md
        assert "| case_key |" in md
        assert "| main |" in md
        assert "| pr |" in md
        assert "| change |" in md
        assert "| status |" in md

    def test_priority_sort_puts_regressed_first(self, tmp_path: Path) -> None:
        """Priority sort mode puts regressed cases before others."""
        main_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
            _make_record("bench-z", "case-2", "n=200", {"n": 200}, 1, 0.100),
            _make_record("bench-z", "case-2", "n=200", {"n": 200}, 2, 0.100),
            _make_record("bench-z", "case-2", "n=200", {"n": 200}, 3, 0.100),
        ]
        pr_records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),  # unchanged
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
            _make_record("bench-z", "case-2", "n=200", {"n": 200}, 1, 0.150),  # regressed
            _make_record("bench-z", "case-2", "n=200", {"n": 200}, 2, 0.150),
            _make_record("bench-z", "case-2", "n=200", {"n": 200}, 3, 0.150),
        ]
        _write_jsonl(tmp_path / "main.jsonl", main_records)
        _write_jsonl(tmp_path / "pr.jsonl", pr_records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        md = render_compare_markdown(compare, sort_mode="priority")

        # Find table rows
        lines = md.split("\n")
        table_rows = [line for line in lines if line.startswith("| bench-")]

        # bench-z (regressed) should come before bench-a (unchanged)
        assert len(table_rows) == 2
        assert "bench-z" in table_rows[0]
        assert "bench-a" in table_rows[1]

    def test_top_n_limits_rows(self, tmp_path: Path) -> None:
        """top_n parameter limits the number of displayed rows."""
        records = []
        for i in range(10):
            for j in range(3):
                records.append(
                    _make_record(f"bench-{i}", f"case-{i}", f"n={i}", {"n": i}, j + 1, 0.1)
                )
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        md = render_compare_markdown(compare, top_n=5)

        # Count table data rows
        lines = md.split("\n")
        table_rows = [line for line in lines if line.startswith("| bench-")]
        assert len(table_rows) == 5
        assert "5 more rows not shown" in md

    def test_top_n_zero_shows_all(self, tmp_path: Path) -> None:
        """top_n=0 shows all rows."""
        records = []
        for i in range(5):
            for j in range(3):
                records.append(
                    _make_record(f"bench-{i}", f"case-{i}", f"n={i}", {"n": i}, j + 1, 0.1)
                )
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        md = render_compare_markdown(compare, top_n=0)

        lines = md.split("\n")
        table_rows = [line for line in lines if line.startswith("| bench-")]
        assert len(table_rows) == 5
        assert "more rows not shown" not in md


class TestWriteCompareJson:
    """Tests for write_compare_json function."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """Output is valid JSON matching input dict."""
        records = [
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 1, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 2, 0.100),
            _make_record("bench-a", "case-1", "n=100", {"n": 100}, 3, 0.100),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        compare = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        out_path = tmp_path / "compare.json"

        write_compare_json(out_path, compare)

        assert out_path.exists()
        with out_path.open() as f:
            loaded = json.load(f)

        assert loaded["metadata"] == compare["metadata"]
        assert loaded["counts"] == compare["counts"]
        assert loaded["rows"] == compare["rows"]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if they don't exist."""
        compare = {"metadata": {}, "counts": {}, "rows": []}
        out_path = tmp_path / "subdir" / "nested" / "compare.json"

        write_compare_json(out_path, compare)

        assert out_path.exists()


class TestWriteCompareMd:
    """Tests for write_compare_md function."""

    def test_writes_markdown_content(self, tmp_path: Path) -> None:
        """Output file contains the provided markdown."""
        md_content = "## Test\n\nSome content"
        out_path = tmp_path / "compare.md"

        write_compare_md(out_path, md_content)

        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == md_content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if they don't exist."""
        md_content = "# Test"
        out_path = tmp_path / "subdir" / "nested" / "compare.md"

        write_compare_md(out_path, md_content)

        assert out_path.exists()


class TestComprehensiveFixture:
    """Tests with a comprehensive fixture covering all classification types."""

    @pytest.fixture
    def comprehensive_fixtures(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create main and PR fixtures with all classification types.

        Cases:
        - bench-regressed: 0.100s -> 0.130s (+30% regression)
        - bench-improved: 0.100s -> 0.070s (-30% improvement)
        - bench-unchanged: 0.100s -> 0.102s (+2% within threshold)
        - bench-added: only in PR (0.050s)
        - bench-removed: only in main (0.060s)
        - bench-noisy: high p95/median ratio triggers attention
        """
        main_records = [
            # Regressed case - stable 100ms baseline
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 1, 0.100),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 2, 0.100),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 3, 0.100),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 4, 0.100),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 5, 0.100),
            # Improved case - stable 100ms baseline
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 1, 0.100),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 2, 0.100),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 3, 0.100),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 4, 0.100),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 5, 0.100),
            # Unchanged case - stable 100ms baseline
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 1, 0.100),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 2, 0.100),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 3, 0.100),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 4, 0.100),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 5, 0.100),
            # Removed case - only in main
            _make_record("bench-removed", "case-rm", "x=4", {"x": 4}, 1, 0.060),
            _make_record("bench-removed", "case-rm", "x=4", {"x": 4}, 2, 0.060),
            _make_record("bench-removed", "case-rm", "x=4", {"x": 4}, 3, 0.060),
            _make_record("bench-removed", "case-rm", "x=4", {"x": 4}, 4, 0.060),
            _make_record("bench-removed", "case-rm", "x=4", {"x": 4}, 5, 0.060),
            # Noisy case - stable in main
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 1, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 2, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 3, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 4, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 5, 0.100),
        ]

        pr_records = [
            # Regressed case - 130ms (+30%)
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 1, 0.130),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 2, 0.130),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 3, 0.130),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 4, 0.130),
            _make_record("bench-regressed", "case-r", "x=1", {"x": 1}, 5, 0.130),
            # Improved case - 70ms (-30%)
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 1, 0.070),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 2, 0.070),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 3, 0.070),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 4, 0.070),
            _make_record("bench-improved", "case-i", "x=2", {"x": 2}, 5, 0.070),
            # Unchanged case - 102ms (+2%)
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 1, 0.102),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 2, 0.102),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 3, 0.102),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 4, 0.102),
            _make_record("bench-unchanged", "case-u", "x=3", {"x": 3}, 5, 0.102),
            # Added case - only in PR
            _make_record("bench-added", "case-a", "x=6", {"x": 6}, 1, 0.050),
            _make_record("bench-added", "case-a", "x=6", {"x": 6}, 2, 0.050),
            _make_record("bench-added", "case-a", "x=6", {"x": 6}, 3, 0.050),
            _make_record("bench-added", "case-a", "x=6", {"x": 6}, 4, 0.050),
            _make_record("bench-added", "case-a", "x=6", {"x": 6}, 5, 0.050),
            # Noisy case - high variance causing high p95/median ratio
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 1, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 2, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 3, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 4, 0.100),
            _make_record("bench-noisy", "case-n", "x=5", {"x": 5}, 5, 0.200),  # outlier
        ]

        main_path = tmp_path / "main.jsonl"
        pr_path = tmp_path / "pr.jsonl"
        _write_jsonl(main_path, main_records)
        _write_jsonl(pr_path, pr_records)

        return main_path, pr_path

    def test_all_classifications_detected(
        self, comprehensive_fixtures: tuple[Path, Path]
    ) -> None:
        """All classification types are correctly identified."""
        main_path, pr_path = comprehensive_fixtures

        result = compare_runs(main_path, pr_path, threshold=0.05, outliers="none")

        assert result["counts"]["regressed"] == 1
        assert result["counts"]["improved"] == 1
        assert result["counts"]["unchanged"] == 2  # includes noisy case (same median)
        assert result["counts"]["added"] == 1
        assert result["counts"]["removed"] == 1

    def test_pct_change_computed_correctly(
        self, comprehensive_fixtures: tuple[Path, Path]
    ) -> None:
        """Percent change is computed as (pr - main) / main."""
        main_path, pr_path = comprehensive_fixtures

        result = compare_runs(main_path, pr_path, threshold=0.05, outliers="none")

        rows_by_bench = {r["bench_name"]: r for r in result["rows"]}

        # Regressed: (0.130 - 0.100) / 0.100 = 0.30
        assert rows_by_bench["bench-regressed"]["pct_change"] == pytest.approx(0.30, rel=0.01)

        # Improved: (0.070 - 0.100) / 0.100 = -0.30
        assert rows_by_bench["bench-improved"]["pct_change"] == pytest.approx(-0.30, rel=0.01)

        # Unchanged: (0.102 - 0.100) / 0.100 = 0.02
        assert rows_by_bench["bench-unchanged"]["pct_change"] == pytest.approx(0.02, rel=0.01)

        # Added/removed have None pct_change
        assert rows_by_bench["bench-added"]["pct_change"] is None
        assert rows_by_bench["bench-removed"]["pct_change"] is None

    def test_high_p95_ratio_triggers_attention(
        self, comprehensive_fixtures: tuple[Path, Path]
    ) -> None:
        """High p95/median ratio triggers needs_attention."""
        main_path, pr_path = comprehensive_fixtures

        result = compare_runs(
            main_path,
            pr_path,
            threshold=0.05,
            outliers="none",  # Keep outlier to see p95 effect
            noise_p95_ratio_threshold=1.10,
        )

        rows_by_bench = {r["bench_name"]: r for r in result["rows"]}

        # Noisy case has one outlier (0.200 vs 0.100 median)
        # p95 will be high relative to median
        noisy_row = rows_by_bench["bench-noisy"]
        assert noisy_row["needs_attention"] is True

        # Stable cases should not need attention
        assert rows_by_bench["bench-regressed"]["needs_attention"] is False
        assert rows_by_bench["bench-improved"]["needs_attention"] is False

    def test_markdown_includes_all_classifications(
        self, comprehensive_fixtures: tuple[Path, Path]
    ) -> None:
        """Markdown output includes rows for all classification types."""
        main_path, pr_path = comprehensive_fixtures

        result = compare_runs(main_path, pr_path, threshold=0.05, outliers="none")
        md = render_compare_markdown(result, top_n=0)

        # All bench names should appear
        assert "bench-regressed" in md
        assert "bench-improved" in md
        assert "bench-unchanged" in md
        assert "bench-added" in md
        assert "bench-removed" in md

        # All emojis should appear
        assert "🔴" in md  # regressed
        assert "🟢" in md  # improved
        assert "⚪" in md  # unchanged
        assert "🆕" in md  # added
        assert "🗑️" in md  # removed


class TestJsonDeterminism:
    """Tests for JSON output determinism."""

    def test_json_rows_identical_across_runs(self, tmp_path: Path) -> None:
        """JSON rows are byte-identical across multiple runs (excluding timestamp)."""
        records = [
            _make_record("bench-z", "case-1", "z=1", {"z": 1}, 1, 0.100),
            _make_record("bench-z", "case-1", "z=1", {"z": 1}, 2, 0.100),
            _make_record("bench-z", "case-1", "z=1", {"z": 1}, 3, 0.100),
            _make_record("bench-a", "case-2", "a=2", {"a": 2}, 1, 0.050),
            _make_record("bench-a", "case-2", "a=2", {"a": 2}, 2, 0.050),
            _make_record("bench-a", "case-2", "a=2", {"a": 2}, 3, 0.050),
            _make_record("bench-m", "case-3", "m=3", {"m": 3}, 1, 0.075),
            _make_record("bench-m", "case-3", "m=3", {"m": 3}, 2, 0.075),
            _make_record("bench-m", "case-3", "m=3", {"m": 3}, 3, 0.075),
        ]
        _write_jsonl(tmp_path / "main.jsonl", records)
        _write_jsonl(tmp_path / "pr.jsonl", records)

        # Run compare twice and write to JSON
        result1 = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")
        result2 = compare_runs(tmp_path / "main.jsonl", tmp_path / "pr.jsonl")

        write_compare_json(tmp_path / "compare1.json", result1)
        write_compare_json(tmp_path / "compare2.json", result2)

        # Load and compare (rows should be identical)
        with open(tmp_path / "compare1.json") as f:
            loaded1 = json.load(f)
        with open(tmp_path / "compare2.json") as f:
            loaded2 = json.load(f)

        # Rows should be byte-identical
        assert json.dumps(loaded1["rows"], sort_keys=True) == json.dumps(
            loaded2["rows"], sort_keys=True
        )

        # Row order should be deterministic (sorted by bench_name)
        bench_names = [r["bench_name"] for r in loaded1["rows"]]
        assert bench_names == ["bench-a", "bench-m", "bench-z"]

    def test_shuffled_input_produces_same_output(self, tmp_path: Path) -> None:
        """Even if input records are in different order, output is deterministic."""
        import random

        base_records = [
            _make_record("bench-z", "case-1", "z=1", {"z": 1}, 1, 0.100),
            _make_record("bench-z", "case-1", "z=1", {"z": 1}, 2, 0.100),
            _make_record("bench-z", "case-1", "z=1", {"z": 1}, 3, 0.100),
            _make_record("bench-a", "case-2", "a=2", {"a": 2}, 1, 0.050),
            _make_record("bench-a", "case-2", "a=2", {"a": 2}, 2, 0.050),
            _make_record("bench-a", "case-2", "a=2", {"a": 2}, 3, 0.050),
        ]

        # Write in original order
        _write_jsonl(tmp_path / "main1.jsonl", base_records)
        _write_jsonl(tmp_path / "pr1.jsonl", base_records)

        # Write in shuffled order
        shuffled = base_records.copy()
        random.seed(12345)
        random.shuffle(shuffled)
        _write_jsonl(tmp_path / "main2.jsonl", shuffled)
        _write_jsonl(tmp_path / "pr2.jsonl", shuffled)

        result1 = compare_runs(tmp_path / "main1.jsonl", tmp_path / "pr1.jsonl")
        result2 = compare_runs(tmp_path / "main2.jsonl", tmp_path / "pr2.jsonl")

        # Rows should be identical regardless of input order
        assert result1["rows"] == result2["rows"]
