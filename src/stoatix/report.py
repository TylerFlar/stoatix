"""Markdown report generator for benchmark results.

This module generates comprehensive Markdown reports from benchmark runs,
including metadata, statistics tables, and optional text-based visualizations.
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from stoatix.summarize import (
    load_results,
    summarize_results,
    summarize_to_csv,
)

# Optional matplotlib support
_HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    pass


def generate_report(
    results_jsonl: Path | str,
    format: str = "md",
    out_path: Path | str | None = None,
    outliers: str = "iqr",
    plots: bool = False,
) -> Path:
    """Generate a Markdown report from benchmark results.

    Auto-discovers sibling artifacts (session.json, cases.json, etc.) from
    the results directory.

    Args:
        results_jsonl: Path to results.jsonl file.
        format: Output format (currently only "md" supported).
        out_path: Output file path. If None, writes to report.md in same dir.
        outliers: Outlier filtering method: "iqr" or "none".
        plots: Whether to include text-based histogram plots.

    Returns:
        Path to the written report file.

    Raises:
        ValueError: If format is not supported or outliers is invalid.
        FileNotFoundError: If results_jsonl does not exist.
    """
    if format != "md":
        raise ValueError(f"Unsupported format '{format}'. Only 'md' is supported.")

    if outliers not in ("iqr", "none"):
        raise ValueError(f"Invalid outliers value '{outliers}'. Must be 'iqr' or 'none'.")

    results_path = Path(results_jsonl)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    out_dir = results_path.parent

    # Determine output path
    if out_path is None:
        report_path = out_dir / "report.md"
    else:
        report_path = Path(out_path)

    # Load results and compute summaries
    records = load_results(results_path)
    summaries = summarize_results(records, outlier_method=outliers)  # type: ignore[arg-type]

    # Sort summaries for stable output
    summaries.sort(key=lambda s: (s["bench_name"], s["case_key"], s["case_id"]))

    # Auto-discover sibling artifacts
    session_data = _load_json_safe(out_dir / "session.json")
    _load_json_safe(out_dir / "cases.json")
    _load_yaml_safe(out_dir / "config.resolved.yml")

    # Ensure summary.csv exists
    summary_csv_path = out_dir / "summary.csv"
    if not summary_csv_path.exists():
        summarize_to_csv(results_path, summary_csv_path, outlier_method=outliers)  # type: ignore[arg-type]

    # Build report
    lines: list[str] = []

    # Title section
    _add_title_section(lines, session_data)

    # Quick links
    _add_quick_links(lines, out_dir, results_path)

    # Run metadata
    _add_metadata_section(lines, session_data)

    # Results table
    _add_results_table(lines, summaries)

    # Failures section
    _add_failures_section(lines, summaries)

    # Variability callouts
    _add_variability_section(lines, summaries)

    # Plots (optional)
    generated_plots: dict[str, str] = {}
    if plots:
        plots_dir = out_dir / "plots"
        generated_plots = _generate_plot_images(summaries, plots_dir)
        _add_plots_section(lines, summaries, generated_plots)

    # Footer with definitions
    _add_footer_section(lines, outliers)

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def _load_json_safe(path: Path) -> dict[str, Any] | None:
    """Load JSON file, returning None if it doesn't exist or fails."""
    try:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _load_yaml_safe(path: Path) -> dict[str, Any] | None:
    """Load YAML file, returning None if it doesn't exist or fails."""
    try:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        pass
    return None


def _add_title_section(lines: list[str], session_data: dict[str, Any] | None) -> None:
    """Add title and timestamp to report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# Stoatix Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")

    if session_data:
        suite_id = session_data.get("suite_id")
        if suite_id:
            lines.append(f"**Suite ID:** `{suite_id}`")

    lines.append("")


def _add_quick_links(lines: list[str], out_dir: Path, results_path: Path) -> None:
    """Add quick links to artifacts."""
    lines.append("## Artifacts")
    lines.append("")

    artifacts = [
        ("results.jsonl", out_dir / "results.jsonl"),
        ("summary.csv", out_dir / "summary.csv"),
        ("session.json", out_dir / "session.json"),
        ("cases.json", out_dir / "cases.json"),
        ("config.resolved.yml", out_dir / "config.resolved.yml"),
    ]

    for name, path in artifacts:
        if path.exists():
            lines.append(f"- [{name}]({name})")

    lines.append("")


def _add_metadata_section(lines: list[str], session_data: dict[str, Any] | None) -> None:
    """Add run metadata section."""
    lines.append("## Run Metadata")
    lines.append("")

    if not session_data:
        lines.append("*Metadata not available (session.json missing)*")
        lines.append("")
        return

    # Git info
    git_info = session_data.get("git_info", {})
    if git_info:
        commit = git_info.get("commit")
        branch = git_info.get("branch")
        is_dirty = git_info.get("is_dirty")

        git_parts = []
        if commit:
            git_parts.append(f"`{commit[:12]}`")
        if branch:
            git_parts.append(f"on `{branch}`")
        if is_dirty:
            git_parts.append("(dirty)")

        if git_parts:
            lines.append(f"**Git:** {' '.join(git_parts)}")

    # System info
    sys_info = session_data.get("system_info", {})
    if sys_info:
        # OS/kernel
        os_info = sys_info.get("os", {})
        kernel_info = sys_info.get("kernel", {})
        os_name = os_info.get("name", "Unknown")
        os_info.get("version", "")
        kernel_release = kernel_info.get("release", "")

        if os_name:
            os_str = os_name
            if kernel_release:
                os_str += f" ({kernel_release})"
            lines.append(f"**OS:** {os_str}")

        # CPU
        cpu_info = sys_info.get("cpu", {})
        cpu_model = cpu_info.get("model")
        logical_cores = cpu_info.get("logical_cores")

        if cpu_model or logical_cores:
            cpu_str = cpu_model or "Unknown CPU"
            if logical_cores:
                cpu_str += f" ({logical_cores} cores)"
            lines.append(f"**CPU:** {cpu_str}")

        # RAM
        ram_bytes = sys_info.get("ram_bytes")
        if ram_bytes:
            ram_str = _humanize_bytes(ram_bytes)
            lines.append(f"**RAM:** {ram_str}")

        # Python
        python_info = sys_info.get("python", {})
        py_version = python_info.get("version")
        py_impl = python_info.get("implementation")
        py_exec = python_info.get("executable")

        if py_version:
            py_str = f"{py_impl} {py_version}" if py_impl else py_version
            lines.append(f"**Python:** {py_str}")
            if py_exec:
                lines.append(f"**Executable:** `{py_exec}`")

        # Command line
        argv = sys_info.get("argv")
        if argv:
            argv_str = " ".join(argv)
            lines.append(f"**Command:** `{argv_str}`")

        # Working directory
        working_dir = sys_info.get("working_dir")
        if working_dir:
            lines.append(f"**Working Dir:** `{working_dir}`")

    lines.append("")


def _humanize_bytes(n: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def _add_results_table(lines: list[str], summaries: list[dict[str, Any]]) -> None:
    """Add main results table."""
    lines.append("## Results")
    lines.append("")

    if not summaries:
        lines.append("*No results to display*")
        lines.append("")
        return

    # Table header
    headers = [
        "bench_name",
        "case_key",
        "n_ok/n_total",
        "median_s",
        "mean_s",
        "stdev_s",
        "p95_s",
        "min_s",
        "max_s",
        "outliers_dropped",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Table rows
    for s in summaries:
        n_ok = s["n_ok"]
        n_total = s["n_total_iterations"]

        row = [
            s["bench_name"],
            s["case_key"] or "-",
            f"{n_ok}/{n_total}",
            _fmt_float(s["median_s"]),
            _fmt_float(s["mean_s"]),
            _fmt_float(s["stdev_s"]),
            _fmt_float(s["p95_s"]),
            _fmt_float(s["min_s"]),
            _fmt_float(s["max_s"]),
            str(s["n_outliers_dropped"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")


def _fmt_float(value: float | None, precision: int = 6) -> str:
    """Format float for display, or '-' if None."""
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _add_failures_section(lines: list[str], summaries: list[dict[str, Any]]) -> None:
    """Add failures section listing cases with failed iterations."""
    failures = [s for s in summaries if s["n_failed"] > 0]

    lines.append("## Failures")
    lines.append("")

    if not failures:
        lines.append("*No failed iterations*")
        lines.append("")
        return

    lines.append(f"**{len(failures)} case(s) with failed iterations:**")
    lines.append("")

    headers = ["bench_name", "case_key", "n_failed", "n_total"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for s in failures:
        row = [
            s["bench_name"],
            s["case_key"] or "-",
            str(s["n_failed"]),
            str(s["n_total_iterations"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")


def _add_variability_section(lines: list[str], summaries: list[dict[str, Any]]) -> None:
    """Add variability callouts section."""
    lines.append("## Variability")
    lines.append("")

    # Filter to cases with stdev available
    with_stdev = [s for s in summaries if s["stdev_s"] is not None]

    if not with_stdev:
        lines.append("*Insufficient data for variability analysis*")
        lines.append("")
        return

    # Sort by stdev descending, take top 5
    top_variable = sorted(with_stdev, key=lambda s: s["stdev_s"], reverse=True)[:5]

    lines.append("**Top 5 cases by standard deviation:**")
    lines.append("")

    headers = ["bench_name", "case_key", "stdev_s", "median_s", "p95_s"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for s in top_variable:
        row = [
            s["bench_name"],
            s["case_key"] or "-",
            _fmt_float(s["stdev_s"]),
            _fmt_float(s["median_s"]),
            _fmt_float(s["p95_s"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces unsafe characters with underscores and limits length.
    """
    # Replace unsafe chars with underscore
    safe = re.sub(r'[<>:"/\\|?*\s]', '_', name)
    # Remove consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Strip leading/trailing underscores
    safe = safe.strip('_')
    # Limit length
    if len(safe) > 100:
        safe = safe[:100]
    return safe or "unnamed"


def _generate_plot_images(
    summaries: list[dict[str, Any]],
    plots_dir: Path,
) -> dict[str, str]:
    """Generate matplotlib boxplot images per benchmark.

    Args:
        summaries: List of summary dictionaries.
        plots_dir: Directory to write plot images.

    Returns:
        Dict mapping bench_name to image path (relative to plots_dir.parent).
    """
    if not _HAS_MATPLOTLIB:
        return {}

    # Group summaries by bench_name
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_bench[s["bench_name"]].append(s)

    generated: dict[str, str] = {}

    for bench_name, bench_summaries in by_bench.items():
        # Collect data for boxplot
        data: list[list[float]] = []
        labels: list[str] = []

        for s in bench_summaries:
            elapsed = s.get("elapsed_values", [])
            if elapsed:
                data.append(elapsed)
                labels.append(s["case_key"] or "(no params)")

        if not data:
            continue

        # Create plot
        try:
            fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.5), 5))
            ax.boxplot(data, labels=labels, vert=True)
            ax.set_title(f"{bench_name} - Elapsed Time Distribution")
            ax.set_ylabel("elapsed_s")
            ax.set_xlabel("case")

            # Rotate labels if many cases
            if len(labels) > 3:
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            # Save plot
            plots_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _sanitize_filename(bench_name)
            image_path = plots_dir / f"{safe_name}.png"
            fig.savefig(image_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Store relative path from out_dir (use forward slashes for markdown)
            generated[bench_name] = f"plots/{safe_name}.png"

        except Exception:
            # If plotting fails, continue without the image
            plt.close('all')
            continue

    return generated


def _add_plots_section(
    lines: list[str],
    summaries: list[dict[str, Any]],
    generated_plots: dict[str, str],
) -> None:
    """Add distribution plots section with images and text histograms."""
    lines.append("## Distribution Plots")
    lines.append("")

    if not summaries:
        lines.append("*No data for plots*")
        lines.append("")
        return

    # Group summaries by bench_name for organized display
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        by_bench[s["bench_name"]].append(s)

    for bench_name in sorted(by_bench.keys()):
        bench_summaries = by_bench[bench_name]

        lines.append(f"### {bench_name}")
        lines.append("")

        # Link to image if available
        if bench_name in generated_plots:
            image_path = generated_plots[bench_name]
            lines.append(f"![{bench_name} boxplot]({image_path})")
            lines.append("")

        # Add text histograms for each case
        for s in bench_summaries:
            elapsed = s.get("elapsed_values", [])
            if not elapsed or len(elapsed) < 2:
                continue

            case_key = s["case_key"] or "(no params)"
            lines.append(f"**{case_key}** (n={len(elapsed)})")
            lines.append("")
            lines.append("```")
            lines.append(_text_histogram(elapsed))
            lines.append("```")
            lines.append("")


def _text_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    """Generate a simple text histogram.

    Args:
        values: List of numeric values.
        bins: Number of bins.
        width: Maximum bar width in characters.

    Returns:
        Multi-line string with histogram.
    """
    if not values:
        return "(no data)"

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return f"All values: {min_val:.6f}"

    # Compute bin edges
    bin_width = (max_val - min_val) / bins
    bin_counts = [0] * bins

    for v in values:
        idx = int((v - min_val) / bin_width)
        if idx >= bins:
            idx = bins - 1
        bin_counts[idx] += 1

    max_count = max(bin_counts) if bin_counts else 1

    # Build histogram
    lines = []
    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"{bin_start:10.6f} - {bin_end:10.6f} | {bar} ({count})")

    return "\n".join(lines)


def _add_footer_section(lines: list[str], outliers: str) -> None:
    """Add footer with definitions and policies."""
    lines.append("---")
    lines.append("")
    lines.append("## Definitions & Policies")
    lines.append("")
    lines.append("- **Measured runs only**: Warmup iterations are excluded from statistics.")
    lines.append("- **Retry handling**: For each iteration, the first successful attempt (`ok=true`) is used.")
    lines.append("  Failed iterations (all attempts failed) are excluded from timing statistics.")

    if outliers == "iqr":
        lines.append("- **Outlier filtering**: IQR method with 1.5×IQR fences per case.")
        lines.append("  Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are dropped.")
    else:
        lines.append("- **Outlier filtering**: None (all successful elapsed times included).")

    lines.append("- **stdev_s**: Sample standard deviation (ddof=1). Blank if n < 2.")
    lines.append("- **p95_s**: 95th percentile using linear interpolation between adjacent sorted values.")
    lines.append("")
