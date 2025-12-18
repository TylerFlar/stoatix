"""Statistics and summary computation for benchmark results.

This module provides functions to aggregate benchmark results from results.jsonl,
compute statistics (mean, median, stdev, percentiles), and handle outliers.

Perf-derived metrics (cpi, ipc, cache_miss_rate) are aggregated when perf stat
was collected during the benchmark run. These fields will be populated only
when record["metrics"]["derived"] exists in the result records.
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal


# Type aliases
OutlierMethod = Literal["none", "iqr"]


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation.

    Uses the "exclusive" method (also known as R-6 or Excel's PERCENTILE.EXC):
    - Sort the data
    - Compute rank = p * (n + 1)
    - Interpolate between adjacent values

    This method is deterministic and does not require numpy.

    Args:
        values: List of numeric values (must not be empty).
        p: Percentile to compute, in range [0, 1].

    Returns:
        The p-th percentile value.

    Raises:
        ValueError: If values is empty or p is out of range.
    """
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    if not 0 <= p <= 1:
        raise ValueError(f"Percentile p must be in [0, 1], got {p}")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Use linear interpolation method (similar to numpy's 'linear')
    # rank is 0-indexed position in the sorted array
    rank = p * (n - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))

    if lower_idx == upper_idx:
        return sorted_values[lower_idx]

    # Linear interpolation between adjacent values
    fraction = rank - lower_idx
    lower_val = sorted_values[lower_idx]
    upper_val = sorted_values[upper_idx]

    return lower_val + fraction * (upper_val - lower_val)


def iqr_filter(values: list[float], k: float = 1.5) -> tuple[list[float], int]:
    """Filter outliers using the IQR (Interquartile Range) method.

    Computes Q1 (25th percentile) and Q3 (75th percentile), then removes
    values outside [Q1 - k*IQR, Q3 + k*IQR] where IQR = Q3 - Q1.

    Args:
        values: List of numeric values.
        k: IQR multiplier for determining outlier bounds (default 1.5).

    Returns:
        Tuple of (filtered_values, n_outliers_dropped).
    """
    if len(values) < 4:
        # Not enough data points for meaningful IQR filtering
        return values.copy(), 0

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    n_dropped = len(values) - len(filtered)

    return filtered, n_dropped


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load results from a JSONL file.

    Args:
        results_path: Path to results.jsonl file.

    Returns:
        List of result record dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    records: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    """Create a grouping key from a record.

    Groups by (bench_name, case_id, case_key, params_json).
    params is serialized to JSON for hashability.
    """
    params_json = json.dumps(record["params"], sort_keys=True)
    return (record["bench_name"], record["case_id"], record["case_key"], params_json)


def _select_ok_record(
    iteration_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the first ok attempt record for an iteration.

    Records should be pre-sorted by attempt number.

    Args:
        iteration_records: List of records for a single (case_id, iteration).

    Returns:
        First ok attempt record, or None if no ok attempt exists.
    """
    sorted_records = sorted(iteration_records, key=lambda r: r["attempt"])
    for record in sorted_records:
        if record["ok"]:
            return record
    return None


def _compute_stats(values: list[float]) -> dict[str, float | None]:
    """Compute statistics for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Dictionary with median, mean, stdev, p95 keys.
    """
    if not values:
        return {"median": None, "mean": None, "stdev": None, "p95": None}

    result: dict[str, float | None] = {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": None,
        "p95": percentile(values, 0.95),
    }

    if len(values) >= 2:
        result["stdev"] = statistics.stdev(values)

    return result


def summarize_results(
    records: list[dict[str, Any]],
    outlier_method: OutlierMethod = "none",
) -> list[dict[str, Any]]:
    """Compute summary statistics from benchmark result records.

    Args:
        records: List of result records (from results.jsonl).
        outlier_method: Method for filtering outliers:
            - "none": Keep all ok elapsed values.
            - "iqr": Remove values outside [Q1-1.5*IQR, Q3+1.5*IQR].

    Returns:
        List of summary dictionaries, one per case, with keys:
            - bench_name: Benchmark name
            - case_id: Case identifier
            - case_key: Human-readable case key (e.g., "x=1,y=2")
            - params: Parameter dictionary
            - n_total_iterations: Total distinct iterations
            - n_ok: Iterations with at least one successful attempt
            - n_failed: Iterations with no successful attempt
            - n_outliers_dropped: Count of values removed by outlier filter
            - elapsed_values: List of elapsed_s values (after outlier filtering)
            - min_s: Minimum elapsed time (None if no data)
            - max_s: Maximum elapsed time (None if no data)
            - median_s: Median elapsed time (None if no data)
            - mean_s: Mean elapsed time (None if no data)
            - stdev_s: Sample standard deviation (None if n < 2)
            - p95_s: 95th percentile elapsed time (None if no data)
            - median_cpi, mean_cpi, stdev_cpi, p95_cpi: CPI stats (perf stat)
            - median_ipc, mean_ipc, stdev_ipc, p95_ipc: IPC stats (perf stat)
            - median_cache_miss_rate, mean_cache_miss_rate, stdev_cache_miss_rate,
              p95_cache_miss_rate: Cache miss rate stats (perf stat)
    """
    # Filter to measured runs only
    measured = [r for r in records if r.get("run_kind") == "measured"]

    # Group records by case
    case_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in measured:
        key = _group_key(record)
        case_groups[key].append(record)

    summaries: list[dict[str, Any]] = []

    for (bench_name, case_id, case_key, params_json), case_records in case_groups.items():
        params = json.loads(params_json)

        # Group by iteration within this case
        iteration_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in case_records:
            iteration_groups[record["iteration"]].append(record)

        # For each iteration, select the first ok attempt
        elapsed_values: list[float] = []
        cpi_values: list[float] = []
        ipc_values: list[float] = []
        cache_miss_rate_values: list[float] = []
        n_ok = 0
        n_failed = 0

        for iteration in sorted(iteration_groups.keys()):
            ok_record = _select_ok_record(iteration_groups[iteration])
            if ok_record is not None:
                elapsed_values.append(ok_record["elapsed_s"])
                n_ok += 1

                # Extract perf-derived metrics if available
                metrics = ok_record.get("metrics")
                if metrics and isinstance(metrics, dict):
                    derived = metrics.get("derived")
                    if derived and isinstance(derived, dict):
                        if derived.get("cpi") is not None:
                            cpi_values.append(derived["cpi"])
                        if derived.get("ipc") is not None:
                            ipc_values.append(derived["ipc"])
                        if derived.get("cache_miss_rate") is not None:
                            cache_miss_rate_values.append(derived["cache_miss_rate"])
            else:
                n_failed += 1

        n_total = n_ok + n_failed

        # Apply outlier filtering to elapsed values
        n_outliers_dropped = 0
        if outlier_method == "iqr" and elapsed_values:
            elapsed_values, n_outliers_dropped = iqr_filter(elapsed_values)
        # For "none", keep elapsed_values as-is

        # Note: We don't apply outlier filtering to perf metrics since they
        # correlate with elapsed time and filtering would misalign the data

        # Compute elapsed time statistics
        min_s: float | None = None
        max_s: float | None = None
        median_s: float | None = None
        mean_s: float | None = None
        stdev_s: float | None = None
        p95_s: float | None = None

        if elapsed_values:
            min_s = min(elapsed_values)
            max_s = max(elapsed_values)
            median_s = statistics.median(elapsed_values)
            mean_s = statistics.mean(elapsed_values)
            p95_s = percentile(elapsed_values, 0.95)

            if len(elapsed_values) >= 2:
                stdev_s = statistics.stdev(elapsed_values)  # ddof=1 by default

        # Compute perf-derived metric statistics
        cpi_stats = _compute_stats(cpi_values)
        ipc_stats = _compute_stats(ipc_values)
        cache_miss_rate_stats = _compute_stats(cache_miss_rate_values)

        summaries.append({
            "bench_name": bench_name,
            "case_id": case_id,
            "case_key": case_key,
            "params": params,
            "n_total_iterations": n_total,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "n_outliers_dropped": n_outliers_dropped,
            "elapsed_values": elapsed_values,
            "min_s": min_s,
            "max_s": max_s,
            "median_s": median_s,
            "mean_s": mean_s,
            "stdev_s": stdev_s,
            "p95_s": p95_s,
            # CPI stats (populated only when perf stat was collected)
            "median_cpi": cpi_stats["median"],
            "mean_cpi": cpi_stats["mean"],
            "stdev_cpi": cpi_stats["stdev"],
            "p95_cpi": cpi_stats["p95"],
            # IPC stats (populated only when perf stat was collected)
            "median_ipc": ipc_stats["median"],
            "mean_ipc": ipc_stats["mean"],
            "stdev_ipc": ipc_stats["stdev"],
            "p95_ipc": ipc_stats["p95"],
            # Cache miss rate stats (populated only when perf stat was collected)
            "median_cache_miss_rate": cache_miss_rate_stats["median"],
            "mean_cache_miss_rate": cache_miss_rate_stats["mean"],
            "stdev_cache_miss_rate": cache_miss_rate_stats["stdev"],
            "p95_cache_miss_rate": cache_miss_rate_stats["p95"],
        })

    # Sort summaries by bench_name, then case_key for stable output
    summaries.sort(key=lambda s: (s["bench_name"], s["case_key"]))

    return summaries


def summarize_from_file(
    results_path: Path | str,
    outlier_method: OutlierMethod = "none",
) -> list[dict[str, Any]]:
    """Load results from file and compute summaries.

    Convenience function that combines load_results and summarize_results.

    Args:
        results_path: Path to results.jsonl file.
        outlier_method: Method for filtering outliers ("none" or "iqr").

    Returns:
        List of summary dictionaries.
    """
    path = Path(results_path)
    records = load_results(path)
    return summarize_results(records, outlier_method=outlier_method)


# CSV column order (stable, explicit)
CSV_COLUMNS = [
    "bench_name",
    "case_id",
    "case_key",
    "params_json",
    "n_total",
    "n_ok",
    "n_failed",
    "n_outliers_dropped",
    "median_s",
    "mean_s",
    "stdev_s",
    "p95_s",
    "min_s",
    "max_s",
    # Perf-derived metrics (populated only when perf stat was collected)
    "median_cpi",
    "mean_cpi",
    "stdev_cpi",
    "p95_cpi",
    "median_ipc",
    "mean_ipc",
    "stdev_ipc",
    "p95_ipc",
    "median_cache_miss_rate",
    "mean_cache_miss_rate",
    "stdev_cache_miss_rate",
    "p95_cache_miss_rate",
]


def _format_float(value: float | None, precision: int = 6) -> str:
    """Format a float value with fixed precision, or empty string if None.

    Args:
        value: Float value or None.
        precision: Number of decimal places.

    Returns:
        Formatted string or empty string.
    """
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def _summary_to_csv_row(summary: dict[str, Any]) -> dict[str, str]:
    """Convert a summary dict to a CSV row dict with string values.

    Args:
        summary: Summary dictionary from summarize_results.

    Returns:
        Dictionary with CSV column names as keys and string values.
    """
    params_json = json.dumps(summary["params"], sort_keys=True, separators=(",", ":"))

    return {
        "bench_name": summary["bench_name"],
        "case_id": summary["case_id"],
        "case_key": summary["case_key"],
        "params_json": params_json,
        "n_total": str(summary["n_total_iterations"]),
        "n_ok": str(summary["n_ok"]),
        "n_failed": str(summary["n_failed"]),
        "n_outliers_dropped": str(summary["n_outliers_dropped"]),
        "median_s": _format_float(summary["median_s"]),
        "mean_s": _format_float(summary["mean_s"]),
        "stdev_s": _format_float(summary["stdev_s"]),
        "p95_s": _format_float(summary["p95_s"]),
        "min_s": _format_float(summary["min_s"]),
        "max_s": _format_float(summary["max_s"]),
        # Perf-derived metrics
        "median_cpi": _format_float(summary.get("median_cpi")),
        "mean_cpi": _format_float(summary.get("mean_cpi")),
        "stdev_cpi": _format_float(summary.get("stdev_cpi")),
        "p95_cpi": _format_float(summary.get("p95_cpi")),
        "median_ipc": _format_float(summary.get("median_ipc")),
        "mean_ipc": _format_float(summary.get("mean_ipc")),
        "stdev_ipc": _format_float(summary.get("stdev_ipc")),
        "p95_ipc": _format_float(summary.get("p95_ipc")),
        "median_cache_miss_rate": _format_float(summary.get("median_cache_miss_rate")),
        "mean_cache_miss_rate": _format_float(summary.get("mean_cache_miss_rate")),
        "stdev_cache_miss_rate": _format_float(summary.get("stdev_cache_miss_rate")),
        "p95_cache_miss_rate": _format_float(summary.get("p95_cache_miss_rate")),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary rows to a CSV file.

    Rows are sorted by (bench_name, case_key, case_id) for stability.
    Columns are written in a fixed order defined by CSV_COLUMNS.

    Args:
        path: Output CSV file path.
        rows: List of summary dictionaries from summarize_results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort for stable output
    sorted_rows = sorted(
        rows,
        key=lambda r: (r["bench_name"], r["case_key"], r["case_id"]),
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_rows:
            csv_row = _summary_to_csv_row(row)
            writer.writerow(csv_row)


def summarize_to_csv(
    results_path: Path | str,
    out_csv: Path | str,
    outlier_method: OutlierMethod = "iqr",
) -> list[dict[str, Any]]:
    """Load results, compute summaries, and write to CSV.

    Convenience function that combines loading, summarization, and CSV output.

    Args:
        results_path: Path to results.jsonl file.
        out_csv: Output CSV file path.
        outlier_method: Method for filtering outliers ("none" or "iqr").
            Defaults to "iqr".

    Returns:
        List of summary dictionaries (same as summarize_results output).
    """
    summaries = summarize_from_file(results_path, outlier_method=outlier_method)
    write_summary_csv(Path(out_csv), summaries)
    return summaries
