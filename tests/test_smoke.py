"""Smoke tests for stoatix CLI."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from stoatix import __version__
from stoatix.cli import app

runner = CliRunner()

VALID_CONFIG = """\
benchmarks:
  - name: test-echo
    command: ["echo", "hello"]
    warmups: 0
    runs: 1
"""


def test_version() -> None:
    """Test the --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help() -> None:
    """Test the help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "stoatix" in result.stdout.lower()


def test_validate_valid_config() -> None:
    """Test validate command with a valid config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(VALID_CONFIG)
        f.flush()
        result = runner.invoke(app, ["validate", f.name])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()


def test_validate_missing_config() -> None:
    """Test validate command with a missing config file."""
    result = runner.invoke(app, ["validate", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_validate_invalid_config() -> None:
    """Test validate command with an invalid config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write("benchmarks: []\n")  # Empty benchmarks list is invalid
        f.flush()
        result = runner.invoke(app, ["validate", f.name])
        assert result.exit_code != 0


def test_run_with_config() -> None:
    """Test run command with a valid config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(VALID_CONFIG)
        out_dir = Path(tmpdir) / "out"

        result = runner.invoke(app, ["run", str(config_path), "--out", str(out_dir)])
        assert result.exit_code == 0
        assert out_dir.exists()
