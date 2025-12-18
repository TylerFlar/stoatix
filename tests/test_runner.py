"""Runner behavior tests."""

import json
from pathlib import Path

from typer.testing import CliRunner

from stoatix.cli import app

runner = CliRunner()


# Simple config with 2 benchmarks and a matrix for determinism tests
DETERMINISM_CONFIG = """\
benchmarks:
  - name: bench-a
    command: ["python", "-c", "print('a')"]
    warmups: 0
    runs: 1
  - name: bench-b
    command: ["python", "-c", "print('{x}')"]
    warmups: 0
    runs: 1
    matrix:
      x: [1, 2]
"""

# Config with retries for failure test
RETRY_FAIL_CONFIG = """\
benchmarks:
  - name: always-fail
    command: ["python", "-c", "import sys; sys.exit(1)"]
    warmups: 0
    runs: 1
    retries: 2
"""

# Config with timeout for timeout test
TIMEOUT_CONFIG = """\
benchmarks:
  - name: slow-cmd
    command: ["python", "-c", "import time; time.sleep(2)"]
    warmups: 0
    runs: 1
    timeout_s: 0.1
    retries: 1
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    """Write config content to a temporary file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)
    return config_path


def _read_cases(out_dir: Path) -> list[dict]:
    """Read cases.json and return the cases list."""
    cases_path = out_dir / "cases.json"
    with cases_path.open() as f:
        data = json.load(f)
    return data["cases"]


def _read_results(out_dir: Path) -> list[dict]:
    """Read results.jsonl and return all records."""
    results_path = out_dir / "results.jsonl"
    records = []
    with results_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class TestDeterministicOrder:
    """Test 1: Deterministic order without shuffle."""

    def test_case_order_is_deterministic(self, tmp_path: Path) -> None:
        """Expanding cases twice should yield identical case_id order."""
        config_path = _write_config(tmp_path, DETERMINISM_CONFIG)
        out_dir1 = tmp_path / "out1"
        out_dir2 = tmp_path / "out2"

        # Run dry-run twice
        result1 = runner.invoke(
            app,
            ["run", str(config_path), "--out", str(out_dir1), "--dry-run"],
        )
        assert result1.exit_code == 0, result1.stdout

        result2 = runner.invoke(
            app,
            ["run", str(config_path), "--out", str(out_dir2), "--dry-run"],
        )
        assert result2.exit_code == 0, result2.stdout

        # Extract case_id order from both
        cases1 = _read_cases(out_dir1)
        cases2 = _read_cases(out_dir2)

        ids1 = [c["case_id"] for c in cases1]
        ids2 = [c["case_id"] for c in cases2]

        assert ids1 == ids2, "Case order should be identical without shuffle"

    def test_case_count_matches_expansion(self, tmp_path: Path) -> None:
        """Should have 3 cases: 1 from bench-a, 2 from bench-b matrix."""
        config_path = _write_config(tmp_path, DETERMINISM_CONFIG)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            ["run", str(config_path), "--out", str(out_dir), "--dry-run"],
        )
        assert result.exit_code == 0

        cases = _read_cases(out_dir)
        assert len(cases) == 3  # 1 (bench-a) + 2 (bench-b with x=[1,2])


class TestShuffleStability:
    """Test 2: Shuffle stability with seed."""

    def test_same_seed_same_order(self, tmp_path: Path) -> None:
        """Running with same seed twice should produce identical order."""
        config_path = _write_config(tmp_path, DETERMINISM_CONFIG)
        out_dir1 = tmp_path / "out1"
        out_dir2 = tmp_path / "out2"

        # Run with --shuffle --seed 123 twice
        result1 = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--out",
                str(out_dir1),
                "--dry-run",
                "--shuffle",
                "--seed",
                "123",
            ],
        )
        assert result1.exit_code == 0, result1.stdout

        result2 = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--out",
                str(out_dir2),
                "--dry-run",
                "--shuffle",
                "--seed",
                "123",
            ],
        )
        assert result2.exit_code == 0, result2.stdout

        # Extract case_id order from both
        cases1 = _read_cases(out_dir1)
        cases2 = _read_cases(out_dir2)

        ids1 = [c["case_id"] for c in cases1]
        ids2 = [c["case_id"] for c in cases2]

        assert ids1 == ids2, "Same seed should produce identical order"

    def test_different_seed_different_order(self, tmp_path: Path) -> None:
        """Running with different seeds should (likely) produce different order.

        Note: There's a small chance of collision (1/6 for 3 items), so we
        don't fail if they happen to match, but we check they're different.
        """
        config_path = _write_config(tmp_path, DETERMINISM_CONFIG)
        out_dir1 = tmp_path / "out1"
        out_dir2 = tmp_path / "out2"

        # Run with --shuffle --seed 123
        result1 = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--out",
                str(out_dir1),
                "--dry-run",
                "--shuffle",
                "--seed",
                "123",
            ],
        )
        assert result1.exit_code == 0

        # Run with --shuffle --seed 124
        result2 = runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--out",
                str(out_dir2),
                "--dry-run",
                "--shuffle",
                "--seed",
                "124",
            ],
        )
        assert result2.exit_code == 0

        cases1 = _read_cases(out_dir1)
        cases2 = _read_cases(out_dir2)

        ids1 = [c["case_id"] for c in cases1]
        ids2 = [c["case_id"] for c in cases2]

        # Different seeds should produce different order (with high probability)
        # Allow test to pass on rare collision
        assert ids1 != ids2, (
            "Different seeds should produce different order "
            "(rare collision possible, re-run test if this fails)"
        )

    def test_session_records_shuffle_info(self, tmp_path: Path) -> None:
        """Session metadata should record shuffle_enabled and seed."""
        config_path = _write_config(tmp_path, DETERMINISM_CONFIG)
        out_dir = tmp_path / "out"

        runner.invoke(
            app,
            [
                "run",
                str(config_path),
                "--out",
                str(out_dir),
                "--dry-run",
                "--shuffle",
                "--seed",
                "42",
            ],
        )

        session_path = out_dir / "session.json"
        with session_path.open() as f:
            session = json.load(f)

        assert session["shuffle_enabled"] is True
        assert session["seed"] == 42


class TestRetriesOnFailure:
    """Test 3: Retries on failure."""

    def test_retry_attempts_recorded(self, tmp_path: Path) -> None:
        """Command that always fails should produce (1 + retries) attempt records."""
        config_path = _write_config(tmp_path, RETRY_FAIL_CONFIG)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            ["run", str(config_path), "--out", str(out_dir)],
        )
        # Exit code 0 - suite completes even if benchmarks fail
        assert result.exit_code == 0

        records = _read_results(out_dir)

        # Should have 3 attempts (1 initial + 2 retries)
        assert len(records) == 3

        # Verify attempt numbers are 1, 2, 3
        attempts = [r["attempt"] for r in records]
        assert attempts == [1, 2, 3]

        # All should be for iteration 0, measured
        for r in records:
            assert r["iteration"] == 0
            assert r["run_kind"] == "measured"
            assert r["ok"] is False
            assert r["exit_code"] == 1
            assert r["timed_out"] is False


class TestTimeoutRecords:
    """Test 4: Timeout records."""

    def test_timeout_records_correct(self, tmp_path: Path) -> None:
        """Timed-out command should show timed_out=true and exit_code=null."""
        config_path = _write_config(tmp_path, TIMEOUT_CONFIG)
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            ["run", str(config_path), "--out", str(out_dir)],
        )
        assert result.exit_code == 0

        records = _read_results(out_dir)

        # Should have 2 attempts (1 initial + 1 retry)
        assert len(records) == 2

        for r in records:
            assert r["timed_out"] is True
            assert r["exit_code"] is None
            assert r["ok"] is False
            assert r["bench_name"] == "slow-cmd"

        # Verify attempt numbers
        attempts = [r["attempt"] for r in records]
        assert attempts == [1, 2]

    def test_timeout_elapsed_time(self, tmp_path: Path) -> None:
        """Elapsed time should be approximately the timeout value."""
        config_path = _write_config(tmp_path, TIMEOUT_CONFIG)
        out_dir = tmp_path / "out"

        runner.invoke(
            app,
            ["run", str(config_path), "--out", str(out_dir)],
        )

        records = _read_results(out_dir)

        for r in records:
            # Elapsed should be close to timeout (0.1s) with some tolerance
            assert 0.05 <= r["elapsed_s"] <= 0.5, (
                f"Elapsed time {r['elapsed_s']} should be near timeout 0.1s"
            )
