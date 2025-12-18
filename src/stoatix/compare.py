"""Comparison of two Stoatix benchmark result files.

This module provides functionality to compare benchmark results between
two runs (e.g., main branch vs PR) and identify regressions, improvements,
and cases needing attention.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from stoatix.summarize import OutlierMethod, load_results, summarize_results


def compare_runs(
    main_results: Path | str,
    pr_results: Path | str,
    threshold: float = 0.05,
    outliers: OutlierMethod = "iqr",
    metric: str = "median_s",
    noise_cv_threshold: float = 0.05,
    noise_p95_ratio_threshold: float = 1.10,
    min_ok: int = 3,
) -> dict[str, Any]:
    """Compare two benchmark result files and identify changes.

    Args:
        main_results: Path to the baseline results.jsonl (e.g., main branch).
        pr_results: Path to the comparison results.jsonl (e.g., PR branch).
        threshold: Percentage threshold for classifying changes.
            A change of > threshold is "regressed", < -threshold is "improved".
        outliers: Outlier filtering method: "iqr" or "none".
        metric: The metric to use for comparison (default "median_s").
        noise_cv_threshold: Coefficient of variation threshold for needs_attention.
            CV = stdev_s / median_s. If >= this, case needs attention.
        noise_p95_ratio_threshold: P95/median ratio threshold for needs_attention.
            If p95_s / median_s >= this, case needs attention.
        min_ok: Minimum number of OK iterations. If either run has fewer,
            case needs attention.

    Returns:
        Dict containing:
            - metadata: Comparison parameters and timestamps
            - counts: Summary counts of each classification
            - rows: List of per-case comparison results

    Raises:
        FileNotFoundError: If either results file does not exist.
        ValueError: If metric is not a valid summary field.
    """
    main_path = Path(main_results)
    pr_path = Path(pr_results)

    if not main_path.exists():
        raise FileNotFoundError(f"Main results file not found: {main_path}")
    if not pr_path.exists():
        raise FileNotFoundError(f"PR results file not found: {pr_path}")

    # Load and summarize both result sets
    main_records = load_results(main_path)
    pr_records = load_results(pr_path)

    main_summaries = summarize_results(main_records, outlier_method=outliers)
    pr_summaries = summarize_results(pr_records, outlier_method=outliers)

    # Index summaries by (bench_name, case_id)
    main_index = _index_summaries(main_summaries)
    pr_index = _index_summaries(pr_summaries)

    # Get all unique case keys
    all_case_keys = set(main_index.keys()) | set(pr_index.keys())

    # Compare each case
    rows: list[dict[str, Any]] = []
    counts = {
        "regressed": 0,
        "improved": 0,
        "unchanged": 0,
        "added": 0,
        "removed": 0,
        "needs_attention": 0,
    }

    for case_key in all_case_keys:
        main_summary = main_index.get(case_key)
        pr_summary = pr_index.get(case_key)

        row = _compare_case(
            case_key=case_key,
            main_summary=main_summary,
            pr_summary=pr_summary,
            metric=metric,
            threshold=threshold,
            noise_cv_threshold=noise_cv_threshold,
            noise_p95_ratio_threshold=noise_p95_ratio_threshold,
            min_ok=min_ok,
        )

        rows.append(row)

        # Update counts
        classification = row["classification"]
        counts[classification] += 1
        if row["needs_attention"]:
            counts["needs_attention"] += 1

    # Sort rows for deterministic output
    rows.sort(key=lambda r: (r["bench_name"], r["case_key"], r["case_id"]))

    # Build result
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "metadata": {
            "created_at_utc": created_at,
            "threshold": threshold,
            "outliers": outliers,
            "metric": metric,
            "noise_cv_threshold": noise_cv_threshold,
            "noise_p95_ratio_threshold": noise_p95_ratio_threshold,
            "min_ok": min_ok,
            "main_path": str(main_path),
            "pr_path": str(pr_path),
        },
        "counts": counts,
        "rows": rows,
    }


def _index_summaries(
    summaries: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index summaries by (bench_name, case_id)."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in summaries:
        key = (summary["bench_name"], summary["case_id"])
        index[key] = summary
    return index


def _compare_case(
    case_key: tuple[str, str],
    main_summary: dict[str, Any] | None,
    pr_summary: dict[str, Any] | None,
    metric: str,
    threshold: float,
    noise_cv_threshold: float,
    noise_p95_ratio_threshold: float,
    min_ok: int,
) -> dict[str, Any]:
    """Compare a single case between main and PR.

    Args:
        case_key: Tuple of (bench_name, case_id).
        main_summary: Summary dict from main run, or None if missing.
        pr_summary: Summary dict from PR run, or None if missing.
        metric: The metric field to compare.
        threshold: Percentage threshold for classification.
        noise_cv_threshold: CV threshold for needs_attention.
        noise_p95_ratio_threshold: P95/median ratio threshold.
        min_ok: Minimum OK count threshold.

    Returns:
        Comparison row dict.
    """
    bench_name, case_id = case_key

    # Determine case_key (display string) and params from whichever is available
    if main_summary is not None:
        case_key_str = main_summary["case_key"]
        params = main_summary.get("params", {})
    else:
        # Must have pr_summary
        case_key_str = pr_summary["case_key"]  # type: ignore[index]
        params = pr_summary.get("params", {})  # type: ignore[union-attr]

    # Make params_json deterministic
    params_json = json.dumps(params, sort_keys=True) if params else "{}"

    # Extract values from summaries
    main_metric = _get_metric(main_summary, metric)
    pr_metric = _get_metric(pr_summary, metric)
    main_n_ok = main_summary["n_ok"] if main_summary else None
    pr_n_ok = pr_summary["n_ok"] if pr_summary else None
    main_stdev = main_summary.get("stdev_s") if main_summary else None
    pr_stdev = pr_summary.get("stdev_s") if pr_summary else None
    main_p95 = main_summary.get("p95_s") if main_summary else None
    pr_p95 = pr_summary.get("p95_s") if pr_summary else None
    main_median = main_summary.get("median_s") if main_summary else None
    pr_median = pr_summary.get("median_s") if pr_summary else None

    # Calculate percentage change
    pct_change: float | None = None
    if main_metric is not None and pr_metric is not None and main_metric != 0:
        pct_change = (pr_metric - main_metric) / main_metric

    # Determine classification
    classification: Literal["regressed", "improved", "unchanged", "added", "removed"]

    if main_summary is None:
        classification = "added"
    elif pr_summary is None:
        classification = "removed"
    elif pct_change is not None:
        if pct_change > threshold:
            classification = "regressed"
        elif pct_change < -threshold:
            classification = "improved"
        else:
            classification = "unchanged"
    else:
        # Both present but pct_change is None (e.g., base is 0)
        classification = "unchanged"

    # Determine needs_attention
    needs_attention = _check_needs_attention(
        main_stdev=main_stdev,
        main_median=main_median,
        main_p95=main_p95,
        main_n_ok=main_n_ok,
        pr_stdev=pr_stdev,
        pr_median=pr_median,
        pr_p95=pr_p95,
        pr_n_ok=pr_n_ok,
        noise_cv_threshold=noise_cv_threshold,
        noise_p95_ratio_threshold=noise_p95_ratio_threshold,
        min_ok=min_ok,
    )

    # Build row with metric-named columns
    row: dict[str, Any] = {
        "bench_name": bench_name,
        "case_id": case_id,
        "case_key": case_key_str,
        "params_json": params_json,
        f"main_{metric}": main_metric,
        f"pr_{metric}": pr_metric,
        "pct_change": pct_change,
        "classification": classification,
        "needs_attention": needs_attention,
        "main_n_ok": main_n_ok,
        "pr_n_ok": pr_n_ok,
        "main_stdev_s": main_stdev,
        "pr_stdev_s": pr_stdev,
        "main_p95_s": main_p95,
        "pr_p95_s": pr_p95,
    }

    return row


def _get_metric(summary: dict[str, Any] | None, metric: str) -> float | None:
    """Get metric value from summary, returning None if unavailable."""
    if summary is None:
        return None
    return summary.get(metric)


def _check_needs_attention(
    main_stdev: float | None,
    main_median: float | None,
    main_p95: float | None,
    main_n_ok: int | None,
    pr_stdev: float | None,
    pr_median: float | None,
    pr_p95: float | None,
    pr_n_ok: int | None,
    noise_cv_threshold: float,
    noise_p95_ratio_threshold: float,
    min_ok: int,
) -> bool:
    """Check if a case needs attention due to noise or insufficient data.

    Args:
        main_stdev, main_median, main_p95, main_n_ok: Stats from main run.
        pr_stdev, pr_median, pr_p95, pr_n_ok: Stats from PR run.
        noise_cv_threshold: CV threshold.
        noise_p95_ratio_threshold: P95/median ratio threshold.
        min_ok: Minimum OK count.

    Returns:
        True if case needs attention, False otherwise.
    """
    # Check CV (coefficient of variation) for main
    if main_stdev is not None and main_median is not None and main_median != 0:
        main_cv = main_stdev / main_median
        if main_cv >= noise_cv_threshold:
            return True

    # Check CV for PR
    if pr_stdev is not None and pr_median is not None and pr_median != 0:
        pr_cv = pr_stdev / pr_median
        if pr_cv >= noise_cv_threshold:
            return True

    # Check P95/median ratio for main
    if main_p95 is not None and main_median is not None and main_median != 0:
        main_p95_ratio = main_p95 / main_median
        if main_p95_ratio >= noise_p95_ratio_threshold:
            return True

    # Check P95/median ratio for PR
    if pr_p95 is not None and pr_median is not None and pr_median != 0:
        pr_p95_ratio = pr_p95 / pr_median
        if pr_p95_ratio >= noise_p95_ratio_threshold:
            return True

    # Check minimum OK count
    if main_n_ok is not None and main_n_ok < min_ok:
        return True
    if pr_n_ok is not None and pr_n_ok < min_ok:
        return True

    return False


# Classification priority for "priority" sort mode
_CLASSIFICATION_PRIORITY = {
    "regressed": 0,
    "improved": 1,
    "unchanged": 2,
    "added": 3,
    "removed": 4,
}


def render_compare_markdown(
    compare: dict[str, Any],
    top_n: int = 20,
    sort_mode: Literal["stable", "priority"] = "stable",
) -> str:
    """Render comparison results to Markdown for PR comments.

    Args:
        compare: Comparison dict from compare_runs().
        top_n: Maximum number of rows to show in the table.
            If 0 or negative, show all rows.
        sort_mode: Row ordering mode.
            - "stable": Sort by (bench_name, case_key, case_id).
            - "priority": Sort by classification priority (regressed first),
              then by stable tuple.

    Returns:
        Markdown string ready to paste into a PR comment.
    """
    lines: list[str] = []
    meta = compare["metadata"]
    counts = compare["counts"]
    rows = compare["rows"]

    # Header
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append(
        f"**Metric:** `{meta['metric']}` | "
        f"**Threshold:** {meta['threshold']:.1%} | "
        f"**Outliers:** `{meta['outliers']}`"
    )
    lines.append("")

    # Summary counts
    summary_parts = []
    if counts["regressed"] > 0:
        summary_parts.append(f"🔴 {counts['regressed']} regressed")
    if counts["improved"] > 0:
        summary_parts.append(f"🟢 {counts['improved']} improved")
    if counts["unchanged"] > 0:
        summary_parts.append(f"⚪ {counts['unchanged']} unchanged")
    if counts["added"] > 0:
        summary_parts.append(f"🆕 {counts['added']} added")
    if counts["removed"] > 0:
        summary_parts.append(f"🗑️ {counts['removed']} removed")

    if counts["needs_attention"] > 0:
        summary_parts.append(f"⚠️ {counts['needs_attention']} need attention")

    lines.append(" | ".join(summary_parts) if summary_parts else "No cases to compare")
    lines.append("")

    # Sort rows based on mode
    if sort_mode == "priority":
        sorted_rows = sorted(
            rows,
            key=lambda r: (
                _CLASSIFICATION_PRIORITY.get(r["classification"], 99),
                r["bench_name"],
                r["case_key"],
                r["case_id"],
            ),
        )
    else:
        # "stable" - already sorted by compare_runs
        sorted_rows = rows

    # Limit rows
    if top_n > 0:
        display_rows = sorted_rows[:top_n]
        truncated = len(sorted_rows) - top_n
    else:
        display_rows = sorted_rows
        truncated = 0

    # Table
    if display_rows:
        metric = meta["metric"]
        lines.append(
            "| bench_name | case_key | main | pr | change | status | ⚠️ |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")

        for row in display_rows:
            main_val = row.get(f"main_{metric}")
            pr_val = row.get(f"pr_{metric}")
            pct = row["pct_change"]
            classification = row["classification"]
            attention = row["needs_attention"]

            # Format values
            main_str = _fmt_time(main_val)
            pr_str = _fmt_time(pr_val)
            pct_str = _fmt_pct(pct)

            # Classification emoji
            status_emoji = {
                "regressed": "🔴",
                "improved": "🟢",
                "unchanged": "⚪",
                "added": "🆕",
                "removed": "🗑️",
            }.get(classification, "")

            attention_str = "⚠️" if attention else ""

            lines.append(
                f"| {row['bench_name']} | {row['case_key']} | "
                f"{main_str} | {pr_str} | {pct_str} | "
                f"{status_emoji} | {attention_str} |"
            )

        lines.append("")

    if truncated > 0:
        lines.append(f"*({truncated} more rows not shown)*")
        lines.append("")

    return "\n".join(lines)


def _fmt_time(val: float | None) -> str:
    """Format time value for table display."""
    if val is None:
        return "-"
    return f"{val:.6f}"


def _fmt_pct(val: float | None) -> str:
    """Format percentage change for table display."""
    if val is None:
        return "-"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1%}"


def write_compare_json(path: Path | str, compare: dict[str, Any]) -> None:
    """Write comparison results to a JSON file.

    Args:
        path: Output file path.
        compare: Comparison dict from compare_runs().
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)


def write_compare_md(path: Path | str, md: str) -> None:
    """Write comparison Markdown to a file.

    Args:
        path: Output file path.
        md: Markdown string from render_compare_markdown().
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
