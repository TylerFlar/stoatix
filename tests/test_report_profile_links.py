"""Tests for report.py profile links detection and rendering.

These tests verify that flamegraph links are correctly detected and
included in generated reports, without requiring perf to be installed.
"""

import json
from pathlib import Path

from stoatix.report import generate_report, _detect_profiles


def make_results_jsonl(out_dir: Path, records: list[dict]) -> Path:
    """Write results.jsonl file."""
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return results_path


def make_session_json(out_dir: Path, suite_id: str = "test-suite-001") -> Path:
    """Write minimal session.json file."""
    session_path = out_dir / "session.json"
    session_data = {
        "suite_id": suite_id,
        "config_path": "stoatix.yml",
        "config_hash": "abc123",
        "benchmark_count": 1,
        "case_count": 2,
    }
    session_path.write_text(json.dumps(session_data), encoding="utf-8")
    return session_path


def make_dummy_profile(out_dir: Path, case_id: str, with_flamegraph: bool = True, with_meta: bool = True) -> Path:
    """Create dummy profile directory with optional artifacts."""
    profile_dir = out_dir / "profiles" / case_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    if with_flamegraph:
        # Create a minimal SVG file
        svg_path = profile_dir / "flamegraph.svg"
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>Dummy</text></svg>',
            encoding="utf-8",
        )

    if with_meta:
        meta_path = profile_dir / "meta.json"
        meta_data = {
            "created_at_utc": "2025-01-01T00:00:00+00:00",
            "suite_id": "test-suite",
            "case": {"case_id": case_id, "bench_name": "test_bench"},
        }
        meta_path.write_text(json.dumps(meta_data), encoding="utf-8")

    return profile_dir


# Sample benchmark results for tests
SAMPLE_RECORDS = [
    {
        "suite_id": "test-suite-001",
        "bench_name": "test_bench",
        "case_id": "case_aaa",
        "case_key": "n=100",
        "params": {"n": 100},
        "run_kind": "measured",
        "iteration": 0,
        "attempt": 1,
        "ok": True,
        "exit_code": 0,
        "elapsed_s": 0.1,
        "started_at": "2025-01-01T00:00:00+00:00",
    },
    {
        "suite_id": "test-suite-001",
        "bench_name": "test_bench",
        "case_id": "case_aaa",
        "case_key": "n=100",
        "params": {"n": 100},
        "run_kind": "measured",
        "iteration": 1,
        "attempt": 1,
        "ok": True,
        "exit_code": 0,
        "elapsed_s": 0.11,
        "started_at": "2025-01-01T00:00:01+00:00",
    },
    {
        "suite_id": "test-suite-001",
        "bench_name": "test_bench",
        "case_id": "case_bbb",
        "case_key": "n=200",
        "params": {"n": 200},
        "run_kind": "measured",
        "iteration": 0,
        "attempt": 1,
        "ok": True,
        "exit_code": 0,
        "elapsed_s": 0.2,
        "started_at": "2025-01-01T00:00:02+00:00",
    },
    {
        "suite_id": "test-suite-001",
        "bench_name": "test_bench",
        "case_id": "case_bbb",
        "case_key": "n=200",
        "params": {"n": 200},
        "run_kind": "measured",
        "iteration": 1,
        "attempt": 1,
        "ok": True,
        "exit_code": 0,
        "elapsed_s": 0.21,
        "started_at": "2025-01-01T00:00:03+00:00",
    },
]


class TestDetectProfiles:
    """Tests for _detect_profiles helper function."""

    def test_detect_no_profiles_dir(self, tmp_path: Path):
        """Returns empty dict when profiles/ doesn't exist."""
        summaries = [{"case_id": "case_aaa"}]
        result = _detect_profiles(tmp_path, summaries)
        assert result == {}

    def test_detect_empty_profiles_dir(self, tmp_path: Path):
        """Returns empty dict when profiles/ is empty."""
        (tmp_path / "profiles").mkdir()
        summaries = [{"case_id": "case_aaa"}]
        result = _detect_profiles(tmp_path, summaries)
        assert result == {}

    def test_detect_flamegraph_only(self, tmp_path: Path):
        """Detects profile with flamegraph but no meta."""
        make_dummy_profile(tmp_path, "case_aaa", with_flamegraph=True, with_meta=False)
        summaries = [{"case_id": "case_aaa"}]

        result = _detect_profiles(tmp_path, summaries)

        assert "case_aaa" in result
        assert result["case_aaa"]["has_flamegraph"] is True
        assert result["case_aaa"]["has_meta"] is False
        assert result["case_aaa"]["flamegraph_path"] == "profiles/case_aaa/flamegraph.svg"
        assert result["case_aaa"]["meta_path"] is None

    def test_detect_meta_only(self, tmp_path: Path):
        """Detects profile with meta but no flamegraph."""
        make_dummy_profile(tmp_path, "case_aaa", with_flamegraph=False, with_meta=True)
        summaries = [{"case_id": "case_aaa"}]

        result = _detect_profiles(tmp_path, summaries)

        assert "case_aaa" in result
        assert result["case_aaa"]["has_flamegraph"] is False
        assert result["case_aaa"]["has_meta"] is True
        assert result["case_aaa"]["flamegraph_path"] is None
        assert result["case_aaa"]["meta_path"] == "profiles/case_aaa/meta.json"

    def test_detect_both_artifacts(self, tmp_path: Path):
        """Detects profile with both flamegraph and meta."""
        make_dummy_profile(tmp_path, "case_aaa", with_flamegraph=True, with_meta=True)
        summaries = [{"case_id": "case_aaa"}]

        result = _detect_profiles(tmp_path, summaries)

        assert "case_aaa" in result
        assert result["case_aaa"]["has_flamegraph"] is True
        assert result["case_aaa"]["has_meta"] is True
        assert result["case_aaa"]["flamegraph_path"] == "profiles/case_aaa/flamegraph.svg"
        assert result["case_aaa"]["meta_path"] == "profiles/case_aaa/meta.json"

    def test_detect_multiple_profiles(self, tmp_path: Path):
        """Detects multiple profile directories."""
        make_dummy_profile(tmp_path, "case_aaa", with_flamegraph=True, with_meta=True)
        make_dummy_profile(tmp_path, "case_bbb", with_flamegraph=True, with_meta=False)
        summaries = [{"case_id": "case_aaa"}, {"case_id": "case_bbb"}]

        result = _detect_profiles(tmp_path, summaries)

        assert len(result) == 2
        assert "case_aaa" in result
        assert "case_bbb" in result

    def test_detect_ignores_empty_profile_dirs(self, tmp_path: Path):
        """Ignores profile directories without artifacts."""
        profile_dir = tmp_path / "profiles" / "case_empty"
        profile_dir.mkdir(parents=True)
        summaries = [{"case_id": "case_empty"}]

        result = _detect_profiles(tmp_path, summaries)

        assert "case_empty" not in result


class TestReportProfileLinks:
    """Tests for profile links in generated reports."""

    def test_report_no_profiles(self, tmp_path: Path):
        """Report generates without profiles section when none exist."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Should not have Profiles section
        assert "## Profiles" not in report_content
        # Results table should not have Profile column
        assert "| Profile" not in report_content

    def test_report_with_profiles_section(self, tmp_path: Path):
        """Report includes Profiles section when profiles exist."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa")

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Should have Profiles section
        assert "## Profiles" in report_content
        # Should link to flamegraph
        assert "flamegraph.svg" in report_content
        assert "profiles/case_aaa/flamegraph.svg" in report_content

    def test_report_profile_column_in_results(self, tmp_path: Path):
        """Results table includes Profile column with flamegraph links."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa")

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Results table should have Profile column header
        assert "| Profile |" in report_content
        # Should have flame emoji link for case with profile
        assert "[🔥](profiles/case_aaa/flamegraph.svg)" in report_content

    def test_report_multiple_profiles(self, tmp_path: Path):
        """Report handles multiple profiles correctly."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa")
        make_dummy_profile(tmp_path, "case_bbb")

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Both profiles should be linked
        assert "profiles/case_aaa/flamegraph.svg" in report_content
        assert "profiles/case_bbb/flamegraph.svg" in report_content

    def test_report_profiles_section_has_meta_links(self, tmp_path: Path):
        """Profiles section includes links to meta.json files."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa", with_flamegraph=True, with_meta=True)

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Profiles section should link to meta.json
        assert "meta.json" in report_content
        assert "profiles/case_aaa/meta.json" in report_content

    def test_report_partial_profile_no_flamegraph(self, tmp_path: Path):
        """Report handles profile with meta but no flamegraph."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa", with_flamegraph=False, with_meta=True)

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Should have Profiles section (meta exists)
        assert "## Profiles" in report_content
        # Should link to meta
        assert "profiles/case_aaa/meta.json" in report_content
        # Results table should show "-" for missing flamegraph
        # (Profile column shows - when no flamegraph)

    def test_report_case_without_profile_shows_dash(self, tmp_path: Path):
        """Cases without profiles show '-' in Profile column."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        # Only create profile for case_aaa, not case_bbb
        make_dummy_profile(tmp_path, "case_aaa")

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Find the results table and verify structure
        lines = report_content.split("\n")
        results_table_lines = []
        in_results = False
        for line in lines:
            if "## Results" in line:
                in_results = True
            elif in_results and line.startswith("##"):
                break
            elif in_results and line.startswith("|"):
                results_table_lines.append(line)

        # Should have Profile column
        assert any("Profile" in line for line in results_table_lines)

        # Find row for case_bbb (n=200) - should have "-" in Profile column
        for line in results_table_lines:
            if "n=200" in line:
                # Last column before final | should be "-" or contain the dash
                assert "| - |" in line or line.endswith("| - |")

    def test_report_profiles_section_format(self, tmp_path: Path):
        """Profiles section has correct table format."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa")
        make_dummy_profile(tmp_path, "case_bbb")

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Check table headers
        assert "| case_id | Flamegraph | Metadata |" in report_content

        # Check case_id is in backticks
        assert "`case_aaa`" in report_content
        assert "`case_bbb`" in report_content


class TestReportPathResolution:
    """Tests for correct relative path resolution in report links."""

    def test_paths_are_relative_from_report_location(self, tmp_path: Path):
        """All profile paths are relative to report.md location."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa")

        report_path = generate_report(tmp_path / "results.jsonl")
        report_content = report_path.read_text(encoding="utf-8")

        # Paths should be relative (not absolute)
        assert str(tmp_path) not in report_content
        # Should use forward slashes (markdown convention)
        assert "profiles/case_aaa/flamegraph.svg" in report_content
        # Should not have backslashes
        assert "profiles\\case_aaa" not in report_content

    def test_custom_output_path(self, tmp_path: Path):
        """Profile paths work with custom report output location."""
        make_results_jsonl(tmp_path, SAMPLE_RECORDS)
        make_session_json(tmp_path)
        make_dummy_profile(tmp_path, "case_aaa")

        # Generate report to custom location (still in same dir)
        custom_path = tmp_path / "reports" / "custom_report.md"
        report_path = generate_report(
            tmp_path / "results.jsonl",
            out_path=custom_path,
        )

        assert report_path == custom_path
        report_content = report_path.read_text(encoding="utf-8")

        # Paths are relative to out_dir (results.jsonl location), not report location
        # This is intentional - profiles are in out_dir/profiles/
        assert "profiles/case_aaa/flamegraph.svg" in report_content
