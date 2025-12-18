"""Tests for perf stat CSV parsing.

These tests verify parsing logic without requiring perf to be installed.
"""

from pathlib import Path

import pytest

from stoatix.perfstat import compute_derived, parse_perf_csv


# Sample perf stat -x, output (comma-delimited CSV format)
# Format: value,unit,event-name,run-time,percent,...
SAMPLE_PERF_OUTPUT = """\
# started on Thu Dec 18 10:00:00 2025

1234567890,,cycles,1000000,100.00,,
987654321,,instructions,1000000,100.00,,
12345678,,branches,1000000,100.00,,
123456,,branch-misses,1000000,100.00,,
5000000,,cache-references,1000000,100.00,,
50000,,cache-misses,1000000,100.00,,
<not supported>,,context-switches,0,0.00,,
<not counted>,,cpu-migrations,0,0.00,,
1234,,page-faults,1000000,100.00,,
"""


# Sample with large numbers (perf -x, mode does NOT use thousands separators)
# This tests that the parser handles large integers correctly
SAMPLE_LARGE_NUMBERS = """\
# perf stat output
1234567890,,cycles,1000000,100.00,,
987654321,,instructions,1000000,100.00,,
"""


# Sample with partial/missing data
SAMPLE_PARTIAL = """\
# incomplete run
1000000,,cycles,500000,50.00,,
<not supported>,,instructions,0,0.00,,
"""


# Sample with empty values and unusual formats
SAMPLE_EDGE_CASES = """\
# edge cases
,,empty-value,0,0.00,,
0,,zero-value,1000000,100.00,,
123.456,,float-value,1000000,100.00,,
"""


class TestParsePerfCsv:
    """Tests for parse_perf_csv function."""

    def test_parses_standard_output(self, tmp_path: Path) -> None:
        """Test parsing standard perf stat CSV output."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_PERF_OUTPUT, encoding="utf-8")

        stats, errors = parse_perf_csv(perf_file)

        assert stats["cycles"] == 1234567890
        assert stats["instructions"] == 987654321
        assert stats["branches"] == 12345678
        assert stats["branch-misses"] == 123456
        assert stats["cache-references"] == 5000000
        assert stats["cache-misses"] == 50000
        assert stats["page-faults"] == 1234

    def test_unsupported_becomes_none(self, tmp_path: Path) -> None:
        """Test that <not supported> values become None."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_PERF_OUTPUT, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["context-switches"] is None

    def test_not_counted_becomes_none(self, tmp_path: Path) -> None:
        """Test that <not counted> values become None."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_PERF_OUTPUT, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["cpu-migrations"] is None

    def test_handles_large_numbers(self, tmp_path: Path) -> None:
        """Test parsing large integer values."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_LARGE_NUMBERS, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["cycles"] == 1234567890
        assert stats["instructions"] == 987654321

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        """Test that lines starting with # are skipped."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_PERF_OUTPUT, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        # Should not have any key that looks like a comment
        for key in stats:
            assert not key.startswith("#")

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are handled gracefully."""
        content = "\n\n100,,cycles,1000,100,,\n\n"
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(content, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["cycles"] == 100

    def test_handles_zero_values(self, tmp_path: Path) -> None:
        """Test that zero is parsed correctly (not as None)."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_EDGE_CASES, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["zero-value"] == 0.0

    def test_handles_float_values(self, tmp_path: Path) -> None:
        """Test that float values are parsed correctly."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_EDGE_CASES, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["float-value"] == pytest.approx(123.456)

    def test_empty_value_becomes_none(self, tmp_path: Path) -> None:
        """Test that empty values become None."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_EDGE_CASES, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)

        assert stats["empty-value"] is None

    def test_reports_missing_requested_events(self, tmp_path: Path) -> None:
        """Test that missing requested events are reported in errors."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text("100,,cycles,1000,100,,\n", encoding="utf-8")

        _, errors = parse_perf_csv(perf_file, requested_events=["cycles", "instructions"])

        # Should report that instructions is missing
        assert any("instructions" in e for e in errors)

    def test_nonexistent_file_returns_error(self, tmp_path: Path) -> None:
        """Test that a nonexistent file returns an error."""
        perf_file = tmp_path / "nonexistent.csv"

        stats, errors = parse_perf_csv(perf_file)

        assert stats == {}
        assert len(errors) > 0
        assert any("read" in e.lower() or "file" in e.lower() for e in errors)


class TestComputeDerived:
    """Tests for compute_derived function."""

    def test_computes_cpi_correctly(self) -> None:
        """Test CPI calculation: cycles / instructions."""
        perf_stat = {
            "cycles": 1000000,
            "instructions": 500000,
        }

        derived = compute_derived(perf_stat)

        assert derived["cpi"] == pytest.approx(2.0)

    def test_computes_ipc_correctly(self) -> None:
        """Test IPC calculation: instructions / cycles."""
        perf_stat = {
            "cycles": 1000000,
            "instructions": 2000000,
        }

        derived = compute_derived(perf_stat)

        assert derived["ipc"] == pytest.approx(2.0)

    def test_cpi_ipc_are_inverses(self) -> None:
        """Test that CPI and IPC are mathematical inverses."""
        perf_stat = {
            "cycles": 1234567890,
            "instructions": 987654321,
        }

        derived = compute_derived(perf_stat)

        assert derived["cpi"] is not None
        assert derived["ipc"] is not None
        assert derived["cpi"] * derived["ipc"] == pytest.approx(1.0)

    def test_computes_cache_miss_rate(self) -> None:
        """Test cache miss rate calculation: misses / references."""
        perf_stat = {
            "cache-references": 1000000,
            "cache-misses": 10000,
        }

        derived = compute_derived(perf_stat)

        assert derived["cache_miss_rate"] == pytest.approx(0.01)

    def test_returns_none_for_missing_cycles(self) -> None:
        """Test that CPI/IPC are None when cycles is missing."""
        perf_stat = {
            "instructions": 1000000,
        }

        derived = compute_derived(perf_stat)

        assert derived["cpi"] is None
        assert derived["ipc"] is None

    def test_returns_none_for_missing_instructions(self) -> None:
        """Test that CPI/IPC are None when instructions is missing."""
        perf_stat = {
            "cycles": 1000000,
        }

        derived = compute_derived(perf_stat)

        assert derived["cpi"] is None
        assert derived["ipc"] is None

    def test_returns_none_for_zero_divisor(self) -> None:
        """Test that division by zero returns None, not error."""
        perf_stat = {
            "cycles": 0,
            "instructions": 1000000,
            "cache-references": 0,
            "cache-misses": 100,
        }

        derived = compute_derived(perf_stat)

        # cycles=0 means IPC would divide by zero
        assert derived["ipc"] is None
        # instructions!=0 so CPI should work
        assert derived["cpi"] == pytest.approx(0.0)
        # cache-references=0 means cache_miss_rate would divide by zero
        assert derived["cache_miss_rate"] is None

    def test_handles_none_values_in_input(self) -> None:
        """Test that None values in perf_stat are handled gracefully."""
        perf_stat: dict[str, float | None] = {
            "cycles": None,
            "instructions": 1000000,
            "cache-references": 5000000,
            "cache-misses": None,
        }

        derived = compute_derived(perf_stat)

        assert derived["cpi"] is None
        assert derived["ipc"] is None
        assert derived["cache_miss_rate"] is None

    def test_empty_input_returns_all_none(self) -> None:
        """Test that empty input returns all None values."""
        derived = compute_derived({})

        assert derived["cpi"] is None
        assert derived["ipc"] is None
        assert derived["cache_miss_rate"] is None


class TestIntegration:
    """Integration tests combining parsing and derived computation."""

    def test_parse_and_compute_derived(self, tmp_path: Path) -> None:
        """Test full flow: parse perf output then compute derived metrics."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_PERF_OUTPUT, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)
        derived = compute_derived(stats)

        # Expected CPI = 1234567890 / 987654321 ≈ 1.25
        assert derived["cpi"] == pytest.approx(1234567890 / 987654321)

        # Expected IPC = 987654321 / 1234567890 ≈ 0.80
        assert derived["ipc"] == pytest.approx(987654321 / 1234567890)

        # Expected cache miss rate = 50000 / 5000000 = 0.01
        assert derived["cache_miss_rate"] == pytest.approx(0.01)

    def test_partial_data_computes_available_metrics(self, tmp_path: Path) -> None:
        """Test that partial data still computes available metrics."""
        perf_file = tmp_path / "perf.csv"
        perf_file.write_text(SAMPLE_PARTIAL, encoding="utf-8")

        stats, _ = parse_perf_csv(perf_file)
        derived = compute_derived(stats)

        # cycles present but instructions is <not supported> (None)
        assert stats["cycles"] == 1000000
        assert stats["instructions"] is None

        # CPI/IPC should be None since instructions is None
        assert derived["cpi"] is None
        assert derived["ipc"] is None
