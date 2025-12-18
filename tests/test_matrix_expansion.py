"""Tests for matrix expansion and templating"""

import pytest

from stoatix.config import load_config
from stoatix.plan import expand_suite, render_template


class TestMatrixExpansionDeterminism:
    """Tests for deterministic matrix expansion ordering."""

    def test_matrix_keys_sorted_regardless_of_yaml_order(self, tmp_path):
        """Matrix keys should be sorted alphabetically in case_key, regardless of YAML order."""
        config_file = tmp_path / "config.yaml"
        # Keys in YAML are: z, a, m (not alphabetical)
        config_file.write_text(
            """\
benchmarks:
  - name: key-order
    command: ["echo", "{z}", "{a}", "{m}"]
    matrix:
      z: [1]
      a: [2]
      m: [3]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        assert len(cases) == 1
        case = cases[0]
        # case_key should have keys in sorted order: a, m, z
        assert case.case_key == "a=2,m=3,z=1"

    def test_matrix_values_preserve_yaml_order(self, tmp_path):
        """Matrix values should preserve YAML order for each key."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: value-order
    command: ["echo", "{n}"]
    matrix:
      n: [100, 10, 1000, 1]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        # Should have 4 cases in YAML value order
        assert len(cases) == 4
        values = [c.params["n"] for c in cases]
        assert values == [100, 10, 1000, 1]

    def test_cartesian_product_preserves_value_order(self, tmp_path):
        """Cartesian product should iterate keys alphabetically, values in YAML order."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: cartesian
    command: ["echo", "{x}", "{y}"]
    matrix:
      y: [1, 2]
      x: [a, b]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        # Keys sorted: x, y
        # First key (x) is outer loop, second key (y) is inner loop
        # Expected order: (a,1), (a,2), (b,1), (b,2)
        assert len(cases) == 4
        case_keys = [c.case_key for c in cases]
        assert case_keys == ["x=a,y=1", "x=a,y=2", "x=b,y=1", "x=b,y=2"]


class TestCaseIdStability:
    """Tests for case_id determinism and stability."""

    def test_same_bench_params_same_case_id(self, tmp_path):
        """Same benchmark name and params should produce same case_id."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: stable-id
    command: ["echo", "{n}"]
    matrix:
      n: [42]
"""
        )

        config = load_config(config_file)

        # Run expansion twice
        cases1 = expand_suite(config)
        cases2 = expand_suite(config)

        assert cases1[0].case_id == cases2[0].case_id

    def test_case_id_differs_for_different_params(self, tmp_path):
        """Different params should produce different case_ids."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: diff-params
    command: ["echo", "{n}"]
    matrix:
      n: [1, 2]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        assert len(cases) == 2
        assert cases[0].case_id != cases[1].case_id

    def test_case_id_differs_for_different_bench_names(self, tmp_path):
        """Same params but different bench names should produce different case_ids."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: bench-a
    command: ["echo", "a"]
  - name: bench-b
    command: ["echo", "b"]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        assert len(cases) == 2
        assert cases[0].case_id != cases[1].case_id

    def test_case_id_independent_of_command_or_other_fields(self, tmp_path):
        """case_id should only depend on bench name and params, not command/runs/etc."""
        # Create two configs with same name/params but different commands
        config1 = tmp_path / "config1.yaml"
        config1.write_text(
            """\
benchmarks:
  - name: same-name
    command: ["echo", "command1"]
    runs: 5
"""
        )

        config2 = tmp_path / "config2.yaml"
        config2.write_text(
            """\
benchmarks:
  - name: same-name
    command: ["echo", "different-command"]
    runs: 10
    warmups: 3
"""
        )

        cases1 = expand_suite(load_config(config1))
        cases2 = expand_suite(load_config(config2))

        # Same name, no params -> same case_id
        assert cases1[0].case_id == cases2[0].case_id


class TestTemplating:
    """Tests for command templating."""

    def test_missing_placeholder_fails_with_clear_error(self, tmp_path):
        """Command with placeholder not in matrix should fail clearly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: missing-param
    command: ["echo", "{threads}"]
    matrix:
      n: [1, 2]
"""
        )

        config = load_config(config_file)

        with pytest.raises(ValueError) as exc_info:
            expand_suite(config)

        error_msg = str(exc_info.value)
        assert "threads" in error_msg
        # Should indicate this is a templating/placeholder error
        assert "placeholder" in error_msg.lower() or "param" in error_msg.lower() or "missing" in error_msg.lower()

    def test_template_renders_all_params(self, tmp_path):
        """All placeholders should be rendered in the command."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: multi-param
    command: ["prog", "--threads={threads}", "--size={size}"]
    matrix:
      threads: [4]
      size: [100]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        assert len(cases) == 1
        assert cases[0].command == ["prog", "--threads=4", "--size=100"]

    def test_no_matrix_no_templating_needed(self, tmp_path):
        """Benchmark without matrix should work with no placeholders."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """\
benchmarks:
  - name: simple
    command: ["echo", "hello"]
"""
        )

        config = load_config(config_file)
        cases = expand_suite(config)

        assert len(cases) == 1
        assert cases[0].command == ["echo", "hello"]
        assert cases[0].params == {}
        assert cases[0].case_key == ""


class TestRenderTemplate:
    """Unit tests for render_template function."""

    def test_render_simple_placeholder(self):
        """Simple placeholder substitution."""
        result = render_template("value={x}", {"x": 42})
        assert result == "value=42"

    def test_render_multiple_placeholders(self):
        """Multiple placeholders in same string."""
        result = render_template("{a}-{b}-{c}", {"a": 1, "b": 2, "c": 3})
        assert result == "1-2-3"

    def test_render_no_placeholders(self):
        """String with no placeholders passes through."""
        result = render_template("no placeholders here", {"x": 1})
        assert result == "no placeholders here"

    def test_render_missing_param_raises(self):
        """Missing param should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            render_template("{missing}", {"other": 1})

        assert "missing" in str(exc_info.value)

    def test_render_preserves_non_placeholder_braces(self):
        """Literal braces that aren't placeholders should be preserved."""
        # Double braces escape to single brace
        result = render_template("{{literal}}", {"literal": "ignored"})
        assert result == "{literal}"
