"""Tests for summarization correctness."""

import csv
import json
import statistics
from pathlib import Path

import pytest

from stoatix.summarize import (
    CSV_COLUMNS,
    iqr_filter,
    percentile,
    summarize_from_file,
    summarize_results,
    write_summary_csv,
)


def _make_record(
    bench_name: str,
    case_id: str,
    case_key: str,
    params: dict,
    iteration: int,
    attempt: int,
    elapsed_s: float,
    ok: bool,
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
        "attempt": attempt,
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


class TestPercentile:
    """Tests for percentile function."""

    def test_median_odd(self) -> None:
        """Median of odd-length list."""
        assert percentile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_median_even(self) -> None:
        """Median of even-length list interpolates."""
        # [1, 2, 3, 4] -> rank = 0.5 * 3 = 1.5 -> interpolate between index 1 and 2
        result = percentile([1, 2, 3, 4], 0.5)
        assert result == 2.5

    def test_p95(self) -> None:
        """95th percentile with linear interpolation."""
        # [1, 2, 3, 4, 5] -> rank = 0.95 * 4 = 3.8
        # interpolate: 4 + 0.8 * (5 - 4) = 4.8
        result = percentile([1, 2, 3, 4, 5], 0.95)
        assert result == pytest.approx(4.8)

    def test_single_value(self) -> None:
        """Single value returns itself for any percentile."""
        assert percentile([42.0], 0.5) == 42.0
        assert percentile([42.0], 0.95) == 42.0

    def test_empty_raises(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            percentile([], 0.5)


class TestIqrFilter:
    """Tests for IQR outlier filtering."""

    def test_no_outliers(self) -> None:
        """Normal data without outliers."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        filtered, dropped = iqr_filter(values)
        assert filtered == values
        assert dropped == 0

    def test_outlier_removed(self) -> None:
        """Extreme outlier is removed."""
        # Q1=2, Q3=4, IQR=2, bounds=[2-3, 4+3]=[-1, 7]
        # 100 is way outside
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        filtered, dropped = iqr_filter(values)
        assert 100.0 not in filtered
        assert dropped == 1

    def test_small_list_no_filtering(self) -> None:
        """Lists with < 4 elements skip filtering."""
        values = [1.0, 2.0, 100.0]
        filtered, dropped = iqr_filter(values)
        assert filtered == values
        assert dropped == 0


class TestSummarizeResults:
    """Tests for summarize_results with deterministic fixtures."""

    @pytest.fixture
    def sample_records(self) -> list[dict]:
        """Create sample records with known properties.

        Case A (case_id="aaa"):
          - 5 iterations, all ok
          - elapsed times: [1.0, 1.1, 1.2, 1.3, 10.0]  (10.0 is outlier)

        Case B (case_id="bbb"):
          - 4 iterations total
          - iteration 0: ok (elapsed=2.0)
          - iteration 1: fail then ok (elapsed=2.1 from attempt 2)
          - iteration 2: all attempts fail (n_failed=1)
          - iteration 3: ok (elapsed=2.2)
        """
        records = []

        # Case A: 5 iterations, last one is an outlier
        for i, elapsed in enumerate([1.0, 1.1, 1.2, 1.3, 10.0]):
            records.append(
                _make_record(
                    bench_name="bench-a",
                    case_id="aaa",
                    case_key="x=1",
                    params={"x": 1},
                    iteration=i,
                    attempt=1,
                    elapsed_s=elapsed,
                    ok=True,
                )
            )

        # Case B: mixed success/failure
        # iteration 0: ok
        records.append(
            _make_record(
                bench_name="bench-b",
                case_id="bbb",
                case_key="y=2",
                params={"y": 2},
                iteration=0,
                attempt=1,
                elapsed_s=2.0,
                ok=True,
            )
        )

        # iteration 1: fail then ok
        records.append(
            _make_record(
                bench_name="bench-b",
                case_id="bbb",
                case_key="y=2",
                params={"y": 2},
                iteration=1,
                attempt=1,
                elapsed_s=0.5,  # failed attempt
                ok=False,
            )
        )
        records.append(
            _make_record(
                bench_name="bench-b",
                case_id="bbb",
                case_key="y=2",
                params={"y": 2},
                iteration=1,
                attempt=2,
                elapsed_s=2.1,  # successful retry
                ok=True,
            )
        )

        # iteration 2: all fail
        records.append(
            _make_record(
                bench_name="bench-b",
                case_id="bbb",
                case_key="y=2",
                params={"y": 2},
                iteration=2,
                attempt=1,
                elapsed_s=0.1,
                ok=False,
            )
        )
        records.append(
            _make_record(
                bench_name="bench-b",
                case_id="bbb",
                case_key="y=2",
                params={"y": 2},
                iteration=2,
                attempt=2,
                elapsed_s=0.2,
                ok=False,
            )
        )

        # iteration 3: ok
        records.append(
            _make_record(
                bench_name="bench-b",
                case_id="bbb",
                case_key="y=2",
                params={"y": 2},
                iteration=3,
                attempt=1,
                elapsed_s=2.2,
                ok=True,
            )
        )

        # Add a warmup record that should be ignored
        records.append(
            _make_record(
                bench_name="bench-a",
                case_id="aaa",
                case_key="x=1",
                params={"x": 1},
                iteration=0,
                attempt=1,
                elapsed_s=999.0,
                ok=True,
                run_kind="warmup",
            )
        )

        return records

    def test_case_counts(self, sample_records: list[dict]) -> None:
        """Verify n_ok, n_failed, n_total counts."""
        summaries = summarize_results(sample_records, outlier_method="none")

        # Find case A
        case_a = next(s for s in summaries if s["case_id"] == "aaa")
        assert case_a["n_total_iterations"] == 5
        assert case_a["n_ok"] == 5
        assert case_a["n_failed"] == 0

        # Find case B
        case_b = next(s for s in summaries if s["case_id"] == "bbb")
        assert case_b["n_total_iterations"] == 4
        assert case_b["n_ok"] == 3  # iterations 0, 1, 3 ok
        assert case_b["n_failed"] == 1  # iteration 2 failed

    def test_retry_uses_first_ok(self, sample_records: list[dict]) -> None:
        """Verify first successful attempt is used for retries."""
        summaries = summarize_results(sample_records, outlier_method="none")

        case_b = next(s for s in summaries if s["case_id"] == "bbb")
        # iteration 1 had attempt 1 fail (0.5s) and attempt 2 ok (2.1s)
        # Should use 2.1 from the ok attempt
        assert 2.1 in case_b["elapsed_values"]
        # The failed attempt's elapsed (0.5) should NOT be included
        assert 0.5 not in case_b["elapsed_values"]

    def test_warmups_excluded(self, sample_records: list[dict]) -> None:
        """Verify warmup records are excluded."""
        summaries = summarize_results(sample_records, outlier_method="none")

        case_a = next(s for s in summaries if s["case_id"] == "aaa")
        # warmup had elapsed=999.0, should not appear
        assert 999.0 not in case_a["elapsed_values"]

    def test_iqr_outlier_dropped(self, sample_records: list[dict]) -> None:
        """Verify IQR filtering drops the outlier."""
        summaries = summarize_results(sample_records, outlier_method="iqr")

        case_a = next(s for s in summaries if s["case_id"] == "aaa")
        # Original: [1.0, 1.1, 1.2, 1.3, 10.0]
        # 10.0 is a clear outlier
        assert case_a["n_outliers_dropped"] == 1
        assert 10.0 not in case_a["elapsed_values"]
        assert len(case_a["elapsed_values"]) == 4

    def test_none_outlier_keeps_all(self, sample_records: list[dict]) -> None:
        """Verify outlier_method='none' keeps all values."""
        summaries = summarize_results(sample_records, outlier_method="none")

        case_a = next(s for s in summaries if s["case_id"] == "aaa")
        assert case_a["n_outliers_dropped"] == 0
        assert 10.0 in case_a["elapsed_values"]
        assert len(case_a["elapsed_values"]) == 5

    def test_statistics_with_iqr(self, sample_records: list[dict]) -> None:
        """Verify statistics are correct after IQR filtering."""
        summaries = summarize_results(sample_records, outlier_method="iqr")

        case_a = next(s for s in summaries if s["case_id"] == "aaa")
        # After dropping outlier: [1.0, 1.1, 1.2, 1.3]
        expected_values = [1.0, 1.1, 1.2, 1.3]

        assert case_a["min_s"] == pytest.approx(1.0)
        assert case_a["max_s"] == pytest.approx(1.3)
        assert case_a["median_s"] == pytest.approx(statistics.median(expected_values))
        assert case_a["mean_s"] == pytest.approx(statistics.mean(expected_values))
        assert case_a["stdev_s"] == pytest.approx(statistics.stdev(expected_values))
        # p95 of [1.0, 1.1, 1.2, 1.3]: rank = 0.95 * 3 = 2.85
        # interpolate between index 2 (1.2) and 3 (1.3): 1.2 + 0.85 * 0.1 = 1.285
        assert case_a["p95_s"] == pytest.approx(1.285)

    def test_statistics_with_none(self, sample_records: list[dict]) -> None:
        """Verify statistics differ when outlier is included."""
        summaries = summarize_results(sample_records, outlier_method="none")

        case_a = next(s for s in summaries if s["case_id"] == "aaa")
        # Including outlier: [1.0, 1.1, 1.2, 1.3, 10.0]
        expected_values = [1.0, 1.1, 1.2, 1.3, 10.0]

        assert case_a["max_s"] == pytest.approx(10.0)
        assert case_a["mean_s"] == pytest.approx(statistics.mean(expected_values))
        # Mean with outlier should be much higher than without
        assert case_a["mean_s"] > 2.0  # way higher than 1.15 without outlier

    def test_stdev_none_for_single_value(self) -> None:
        """Verify stdev is None when n < 2."""
        records = [
            _make_record(
                bench_name="single",
                case_id="single",
                case_key="",
                params={},
                iteration=0,
                attempt=1,
                elapsed_s=1.0,
                ok=True,
            )
        ]
        summaries = summarize_results(records, outlier_method="none")
        assert summaries[0]["stdev_s"] is None

    def test_failed_iterations_excluded_from_stats(self, sample_records: list[dict]) -> None:
        """Verify failed iterations don't contribute to timing stats."""
        summaries = summarize_results(sample_records, outlier_method="none")

        case_b = next(s for s in summaries if s["case_id"] == "bbb")
        # Only 3 ok iterations: 2.0, 2.1, 2.2
        assert case_b["elapsed_values"] == pytest.approx([2.0, 2.1, 2.2])


class TestWriteSummaryCsv:
    """Tests for CSV output."""

    def test_csv_column_order(self, tmp_path: Path) -> None:
        """Verify CSV has correct columns in correct order."""
        summaries = [
            {
                "bench_name": "test",
                "case_id": "abc123",
                "case_key": "n=1",
                "params": {"n": 1},
                "n_total_iterations": 5,
                "n_ok": 4,
                "n_failed": 1,
                "n_outliers_dropped": 0,
                "elapsed_values": [1.0, 2.0],
                "min_s": 1.0,
                "max_s": 2.0,
                "median_s": 1.5,
                "mean_s": 1.5,
                "stdev_s": 0.707107,
                "p95_s": 1.95,
            }
        ]

        csv_path = tmp_path / "summary.csv"
        write_summary_csv(csv_path, summaries)

        with csv_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == CSV_COLUMNS

    def test_csv_values_formatted(self, tmp_path: Path) -> None:
        """Verify CSV values are correctly formatted."""
        summaries = [
            {
                "bench_name": "test",
                "case_id": "abc123",
                "case_key": "n=1",
                "params": {"n": 1, "z": "hello"},
                "n_total_iterations": 5,
                "n_ok": 4,
                "n_failed": 1,
                "n_outliers_dropped": 2,
                "elapsed_values": [1.0],
                "min_s": 1.0,
                "max_s": 2.0,
                "median_s": 1.5,
                "mean_s": 1.5,
                "stdev_s": None,  # Should be empty string
                "p95_s": 1.95,
            }
        ]

        csv_path = tmp_path / "summary.csv"
        write_summary_csv(csv_path, summaries)

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["bench_name"] == "test"
        assert row["case_id"] == "abc123"
        assert row["params_json"] == '{"n":1,"z":"hello"}'  # compact, sorted
        assert row["n_total"] == "5"
        assert row["n_ok"] == "4"
        assert row["n_failed"] == "1"
        assert row["n_outliers_dropped"] == "2"
        assert row["stdev_s"] == ""  # None becomes empty
        assert row["mean_s"] == "1.500000"  # 6 decimal places


class TestSummarizeFromFile:
    """Tests for file-based summarization."""

    def test_load_and_summarize(self, tmp_path: Path) -> None:
        """Verify summarize_from_file works end-to-end."""
        records = [
            _make_record(
                bench_name="file-test",
                case_id="xyz",
                case_key="k=v",
                params={"k": "v"},
                iteration=i,
                attempt=1,
                elapsed_s=float(i + 1),
                ok=True,
            )
            for i in range(3)
        ]

        results_path = tmp_path / "results.jsonl"
        _write_jsonl(results_path, records)

        summaries = summarize_from_file(results_path, outlier_method="none")

        assert len(summaries) == 1
        assert summaries[0]["bench_name"] == "file-test"
        assert summaries[0]["n_ok"] == 3
        assert summaries[0]["elapsed_values"] == pytest.approx([1.0, 2.0, 3.0])
