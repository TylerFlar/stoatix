"""Tests for runner perf stat fallback behavior.

These tests verify graceful degradation when perf is unavailable,
without requiring perf to be installed.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stoatix.cli import app
import stoatix.perfstat


@pytest.fixture
def simple_config(tmp_path: Path) -> Path:
    """Create a minimal benchmark config for testing."""
    config = tmp_path / "bench.yml"
    config.write_text(
        """\
benchmarks:
  - name: quick-test
    command: ["python", "-c", "print('hello')"]
    runs: 2
    warmups: 0
""",
        encoding="utf-8",
    )
    return config


class TestPerfFallbackNonLinux:
    """Tests for perf fallback when platform is not Linux.
    
    Since we're already on Windows, we don't need to monkeypatch the platform.
    """

    def test_completes_with_perf_stat_on_non_linux(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that run completes when --perf-stat is used on non-Linux."""
        # We're on Windows, so perf should fail gracefully
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        # Run should complete successfully (exit code 0)
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Results file should exist
        results_file = out_dir / "results.jsonl"
        assert results_file.exists()

    def test_perf_ok_false_on_non_linux(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that metrics.perf_ok is False on non-Linux platform."""
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        assert result.exit_code == 0

        # Read results and check metrics
        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        # Check that all records have metrics with perf_ok=False
        for record in records:
            metrics = record.get("metrics", {})
            assert metrics.get("perf_ok") is False

    def test_perf_error_message_on_non_linux(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that perf_error explains the issue on non-Linux."""
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        assert result.exit_code == 0

        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        # Check that perf_error contains meaningful message
        for record in records:
            metrics = record.get("metrics", {})
            perf_error = metrics.get("perf_error")
            assert perf_error is not None
            assert "linux" in perf_error.lower() or "supported" in perf_error.lower()

    def test_elapsed_time_still_captured_on_non_linux(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that elapsed_s is still captured even when perf fails."""
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        assert result.exit_code == 0

        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        # Check that all records have elapsed_s
        for record in records:
            assert "elapsed_s" in record
            assert isinstance(record["elapsed_s"], (int, float))
            assert record["elapsed_s"] >= 0


class TestPerfFallbackNoPerf:
    """Tests for perf fallback when perf executable is not found.
    
    These tests monkeypatch is_supported() and find_perf() to simulate
    being on Linux but without perf installed.
    """

    def test_completes_when_perf_not_found(
        self,
        simple_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that run completes when perf is not on PATH."""
        # Force perfstat module to think we're on Linux but perf not found
        monkeypatch.setattr(stoatix.perfstat, "is_supported", lambda: True)
        monkeypatch.setattr(stoatix.perfstat, "find_perf", lambda: None)

        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        # Run should complete successfully
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Results file should exist
        results_file = out_dir / "results.jsonl"
        assert results_file.exists()

    def test_perf_ok_false_when_perf_not_found(
        self,
        simple_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that metrics.perf_ok is False when perf not found."""
        monkeypatch.setattr(stoatix.perfstat, "is_supported", lambda: True)
        monkeypatch.setattr(stoatix.perfstat, "find_perf", lambda: None)

        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        assert result.exit_code == 0

        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        for record in records:
            metrics = record.get("metrics", {})
            assert metrics.get("perf_ok") is False

    def test_perf_error_explains_not_found(
        self,
        simple_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that perf_error explains perf was not found."""
        monkeypatch.setattr(stoatix.perfstat, "is_supported", lambda: True)
        monkeypatch.setattr(stoatix.perfstat, "find_perf", lambda: None)

        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
            ],
        )

        assert result.exit_code == 0

        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        for record in records:
            metrics = record.get("metrics", {})
            perf_error = metrics.get("perf_error")
            assert perf_error is not None
            assert "not found" in perf_error.lower() or "path" in perf_error.lower()


class TestPerfStrictMode:
    """Tests for --perf-strict mode behavior."""

    def test_perf_strict_fails_on_non_linux(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that --perf-strict fails when perf unavailable."""
        # On Windows, perf is never available
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
                "--perf-strict",
            ],
        )

        # Should fail in strict mode
        assert result.exit_code != 0

    def test_perf_strict_fails_when_perf_not_found(
        self,
        simple_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that --perf-strict fails when perf not on PATH."""
        monkeypatch.setattr(stoatix.perfstat, "is_supported", lambda: True)
        monkeypatch.setattr(stoatix.perfstat, "find_perf", lambda: None)

        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--perf-stat",
                "--perf-strict",
            ],
        )

        # Should fail in strict mode
        assert result.exit_code != 0


class TestNoPerfStatFlag:
    """Tests to verify behavior when --perf-stat is not used."""

    def test_no_metrics_when_perf_stat_disabled(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that no metrics field exists when --no-perf-stat (default)."""
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
                "--no-perf-stat",
            ],
        )

        assert result.exit_code == 0

        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        # When perf-stat is disabled, metrics field should not be present
        for record in records:
            assert "metrics" not in record or record.get("metrics") is None

    def test_elapsed_time_captured_without_perf(
        self,
        simple_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that timing works without --perf-stat."""
        out_dir = tmp_path / "out"
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "run",
                str(simple_config),
                "--out", str(out_dir),
            ],
        )

        assert result.exit_code == 0

        results_file = out_dir / "results.jsonl"
        records = [json.loads(line) for line in results_file.read_text().splitlines()]

        for record in records:
            assert "elapsed_s" in record
            assert record["elapsed_s"] >= 0
