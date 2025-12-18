"""Tests for report generation."""

import json
from pathlib import Path

import pytest

from stoatix.report import generate_report


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
        "suite_id": "test-suite-12345",
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


def _write_json(path: Path, data: dict) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@pytest.fixture
def benchmark_output(tmp_path: Path) -> Path:
    """Create a deterministic benchmark output directory with results."""
    # Create results.jsonl with two benchmarks, multiple iterations
    records = [
        # Benchmark A - case 1: n=100
        _make_record("bench-a", "case-a1", "n=100", {"n": 100}, 1, 0.100),
        _make_record("bench-a", "case-a1", "n=100", {"n": 100}, 2, 0.105),
        _make_record("bench-a", "case-a1", "n=100", {"n": 100}, 3, 0.098),
        # Benchmark A - case 2: n=200
        _make_record("bench-a", "case-a2", "n=200", {"n": 200}, 1, 0.200),
        _make_record("bench-a", "case-a2", "n=200", {"n": 200}, 2, 0.210),
        _make_record("bench-a", "case-a2", "n=200", {"n": 200}, 3, 0.195),
        # Benchmark B - single case
        _make_record("bench-b", "case-b1", "x=1,y=2", {"x": 1, "y": 2}, 1, 0.050),
        _make_record("bench-b", "case-b1", "x=1,y=2", {"x": 1, "y": 2}, 2, 0.052),
        _make_record("bench-b", "case-b1", "x=1,y=2", {"x": 1, "y": 2}, 3, 0.048),
    ]
    _write_jsonl(tmp_path / "results.jsonl", records)

    # Create session.json
    session_data = {
        "suite_id": "test-suite-12345",
        "config_path": "test.yml",
        "config_hash": "abc123def456",
        "benchmark_count": 2,
        "case_count": 3,
        "started_at_utc": "2025-01-01T00:00:00+00:00",
        "shuffle": False,
        "seed": None,
        "system_info": {
            "os": {"name": "TestOS", "version": "1.0"},
            "kernel": {"release": "5.0"},
            "cpu": {"model": "Test CPU", "cores": 4},
            "memory": {"total_bytes": 8589934592},
            "python": {
                "version": "3.14.0",
                "implementation": "CPython",
                "executable": "/usr/bin/python",
            },
        },
        "git_info": {
            "commit": "abc1234567890",
            "branch": "main",
            "is_dirty": False,
        },
        "invocation": {
            "command": "stoatix run test.yml",
            "cwd": "/test",
        },
    }
    _write_json(tmp_path / "session.json", session_data)

    # Create cases.json
    cases_data = [
        {
            "suite_id": "test-suite-12345",
            "case_id": "case-a1",
            "bench_name": "bench-a",
            "case_key": "n=100",
            "params": {"n": 100},
            "command": ["echo", "100"],
        },
        {
            "suite_id": "test-suite-12345",
            "case_id": "case-a2",
            "bench_name": "bench-a",
            "case_key": "n=200",
            "params": {"n": 200},
            "command": ["echo", "200"],
        },
        {
            "suite_id": "test-suite-12345",
            "case_id": "case-b1",
            "bench_name": "bench-b",
            "case_key": "x=1,y=2",
            "params": {"x": 1, "y": 2},
            "command": ["echo", "x y"],
        },
    ]
    _write_json(tmp_path / "cases.json", cases_data)

    return tmp_path


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_report_file_created(self, benchmark_output: Path) -> None:
        """Report file is created at expected path."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        assert report_path.exists()
        assert report_path.name == "report.md"
        assert report_path.parent == benchmark_output

    def test_report_contains_section_headers(self, benchmark_output: Path) -> None:
        """Report contains expected section headers."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Check for main sections
        assert "# Stoatix Report" in content
        assert "## Artifacts" in content
        assert "## Run Metadata" in content
        assert "## Results" in content
        assert "## Failures" in content
        assert "## Variability" in content

    def test_report_contains_artifact_links(self, benchmark_output: Path) -> None:
        """Report contains links to artifact files."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Check for artifact links
        assert "results.jsonl" in content
        assert "session.json" in content
        assert "cases.json" in content

    def test_report_contains_results_table(self, benchmark_output: Path) -> None:
        """Report contains a results table with all cases."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Check for table headers
        assert "| bench_name |" in content
        assert "| case_key |" in content

        # Check for each benchmark row
        assert "| bench-a |" in content
        assert "| bench-b |" in content

        # Check for each case_key
        assert "n=100" in content
        assert "n=200" in content
        assert "x=1,y=2" in content

    def test_report_contains_metadata(self, benchmark_output: Path) -> None:
        """Report contains run metadata from session.json."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Check for metadata values
        assert "test-suite-12345" in content
        assert "TestOS" in content
        assert "Test CPU" in content
        assert "abc1234" in content  # git commit

    def test_report_contains_definitions(self, benchmark_output: Path) -> None:
        """Report contains definitions section explaining terms."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Check for definitions
        assert "## Definitions" in content
        assert "Warmup iterations" in content or "warmup" in content.lower()

    def test_report_output_is_deterministic(self, benchmark_output: Path) -> None:
        """Re-running generate_report produces identical output."""
        results_path = benchmark_output / "results.jsonl"

        # Generate report twice
        report_path1 = generate_report(results_path, plots=False)
        content1 = report_path1.read_text(encoding="utf-8")

        # Remove the generated timestamp line for comparison
        # (it will differ between runs)
        lines1 = [
            line for line in content1.split("\n")
            if not line.startswith("**Generated:**")
        ]

        report_path2 = generate_report(results_path, plots=False)
        content2 = report_path2.read_text(encoding="utf-8")
        lines2 = [
            line for line in content2.split("\n")
            if not line.startswith("**Generated:**")
        ]

        assert lines1 == lines2

    def test_report_custom_output_path(self, benchmark_output: Path) -> None:
        """Report can be written to a custom path."""
        results_path = benchmark_output / "results.jsonl"
        custom_path = benchmark_output / "custom_report.md"

        report_path = generate_report(results_path, out_path=custom_path, plots=False)

        assert report_path == custom_path
        assert custom_path.exists()

    def test_report_with_outliers_none(self, benchmark_output: Path) -> None:
        """Report can be generated with outliers='none'."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, outliers="none", plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Should mention no outlier filtering in definitions
        assert "outlier" in content.lower()

    def test_report_invalid_format_raises(self, benchmark_output: Path) -> None:
        """Invalid format raises ValueError."""
        results_path = benchmark_output / "results.jsonl"

        with pytest.raises(ValueError, match="Unsupported format"):
            generate_report(results_path, format="pdf", plots=False)

    def test_report_invalid_outliers_raises(self, benchmark_output: Path) -> None:
        """Invalid outliers value raises ValueError."""
        results_path = benchmark_output / "results.jsonl"

        with pytest.raises(ValueError, match="Invalid outliers"):
            generate_report(results_path, outliers="invalid", plots=False)

    def test_report_missing_results_raises(self, tmp_path: Path) -> None:
        """Missing results file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            generate_report(tmp_path / "nonexistent.jsonl", plots=False)


class TestReportWithFailures:
    """Tests for report generation with failed iterations."""

    def test_failures_section_lists_failed_cases(self, tmp_path: Path) -> None:
        """Failures section shows cases with failed iterations."""
        records = [
            _make_record("failing-bench", "case-f1", "mode=fail", {"mode": "fail"}, 1, 0.1, ok=True),
            _make_record("failing-bench", "case-f1", "mode=fail", {"mode": "fail"}, 2, 0.1, ok=False),
            _make_record("failing-bench", "case-f1", "mode=fail", {"mode": "fail"}, 3, 0.1, ok=True),
        ]
        _write_jsonl(tmp_path / "results.jsonl", records)

        # Minimal session.json
        _write_json(tmp_path / "session.json", {
            "suite_id": "test",
            "system": {},
            "git": None,
            "invocation": {},
        })

        report_path = generate_report(tmp_path / "results.jsonl", plots=False)
        content = report_path.read_text(encoding="utf-8")

        # Should have failures section with content (not "No failed iterations")
        assert "## Failures" in content
        # The case should show 2/3 OK (1 failed)
        assert "2/3" in content

    def test_no_failures_message(self, benchmark_output: Path) -> None:
        """When no failures, shows appropriate message."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        assert "No failed iterations" in content


class TestReportTableFormatting:
    """Tests for results table formatting."""

    def test_table_row_count_matches_cases(self, benchmark_output: Path) -> None:
        """Table has one data row per case in Results section."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Extract just the Results section (between ## Results and ## Failures)
        results_section = content.split("## Results")[1].split("## Failures")[0]

        # Count table rows (lines starting with | and containing bench name)
        table_rows = [
            line for line in results_section.split("\n")
            if line.startswith("| bench-")
        ]

        # Should have 3 cases: bench-a n=100, bench-a n=200, bench-b x=1,y=2
        assert len(table_rows) == 3

    def test_table_contains_statistics_columns(self, benchmark_output: Path) -> None:
        """Table contains expected statistics columns."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        # Check for statistics column headers
        assert "median_s" in content
        assert "mean_s" in content
        assert "stdev_s" in content
        assert "p95_s" in content
        assert "min_s" in content
        assert "max_s" in content


class TestReportVariabilitySection:
    """Tests for variability section."""

    def test_variability_shows_top_cases(self, benchmark_output: Path) -> None:
        """Variability section shows cases sorted by stdev."""
        results_path = benchmark_output / "results.jsonl"
        report_path = generate_report(results_path, plots=False)

        content = report_path.read_text(encoding="utf-8")

        assert "## Variability" in content
        assert "Top" in content or "standard deviation" in content.lower()
