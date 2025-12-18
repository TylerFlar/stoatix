"""Tests for config validation error messages (Milestone 1)."""

import pytest

from stoatix.config import load_config


def test_unknown_top_level_key_fails_with_path(tmp_path):
    """Unknown top-level key should fail with the key name in the message."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: test
    command: ["echo", "hi"]
unknown_key: value
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    assert "unknown_key" in str(exc_info.value)


def test_unknown_benchmark_field_fails_with_path(tmp_path):
    """Unknown benchmark field should fail with benchmark name/index in message."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: my-bench
    command: ["echo", "hi"]
    bogus_field: 123
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "bogus_field" in error_msg
    # Should indicate which benchmark has the error
    assert "my-bench" in error_msg or "benchmarks[0]" in error_msg


def test_type_mismatch_runs_string_fails_with_path(tmp_path):
    """Type mismatch (runs: "5" instead of int) should fail with field path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: type-test
    command: ["echo", "hi"]
    runs: "5"
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "runs" in error_msg
    # Should mention the type issue
    assert "int" in error_msg.lower() or "integer" in error_msg.lower() or "type" in error_msg.lower()


def test_type_mismatch_warmups_string_fails(tmp_path):
    """Type mismatch for warmups should also fail."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: warmup-test
    command: ["echo", "hi"]
    warmups: "2"
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "warmups" in error_msg


def test_duplicate_benchmark_names_fail(tmp_path):
    """Duplicate benchmark names should fail with a clear error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: duplicate-name
    command: ["echo", "first"]
  - name: other-bench
    command: ["echo", "middle"]
  - name: duplicate-name
    command: ["echo", "second"]
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "duplicate-name" in error_msg
    assert "duplicate" in error_msg.lower()


def test_unknown_defaults_field_fails(tmp_path):
    """Unknown field in defaults section should fail."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
defaults:
  runs: 5
  invalid_default: true
benchmarks:
  - name: test
    command: ["echo", "hi"]
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "invalid_default" in error_msg


def test_command_must_be_list(tmp_path):
    """Command must be a list, not a string."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: string-cmd
    command: "echo hi"
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "command" in error_msg
    assert "list" in error_msg.lower()


def test_matrix_must_be_dict(tmp_path):
    """Matrix must be a dict, not a list."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: bad-matrix
    command: ["echo", "hi"]
    matrix:
      - n: 1
      - n: 2
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "matrix" in error_msg


def test_matrix_values_must_be_lists(tmp_path):
    """Matrix values must be lists."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmarks:
  - name: bad-matrix-value
    command: ["echo", "{n}"]
    matrix:
      n: 5
"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_file)

    error_msg = str(exc_info.value)
    assert "matrix" in error_msg or "list" in error_msg.lower()
