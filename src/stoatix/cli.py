"""Stoatix CLI entry point."""

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from stoatix import __version__

app = typer.Typer(
    name="stoatix",
    help="A lightweight CLI tool for system benchmarking and diagnostics.",
    add_completion=False,
)

logger = logging.getLogger("stoatix")


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"stoatix {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Print version and exit.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose (DEBUG) logging.",
        ),
    ] = False,
) -> None:
    """Stoatix - A lightweight CLI tool for system benchmarking and diagnostics."""
    _setup_logging(verbose)


@app.command()
def validate(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the configuration file (canonical: stoatix.yml).",
            exists=True,
            readable=True,
        ),
    ],
) -> None:
    """Validate a configuration file and show expansion summary."""
    from stoatix.config import load_config
    from stoatix.plan import expand_suite

    try:
        config = load_config(config_path)
        cases = expand_suite(config)

        # Count cases per benchmark
        cases_per_bench: dict[str, int] = {}
        for case in cases:
            cases_per_bench[case.bench_name] = cases_per_bench.get(case.bench_name, 0) + 1

        typer.echo(f"Configuration: {config_path}")
        typer.echo(f"Benchmarks: {len(config.benchmarks)}")
        typer.echo(f"Total cases: {len(cases)}")
        typer.echo()
        for bench in config.benchmarks:
            count = cases_per_bench.get(bench.name, 0)
            typer.echo(f"  {bench.name}: {count} case(s)")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the configuration file.",
            exists=True,
            readable=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for results.",
        ),
    ] = Path("out"),
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle/--no-shuffle",
            help="Shuffle case execution order for randomized testing.",
        ),
    ] = False,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed for shuffling. Auto-generated if shuffle is enabled without seed.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate and write metadata files without executing benchmarks.",
        ),
    ] = False,
    do_summarize: Annotated[
        bool,
        typer.Option(
            "--summarize/--no-summarize",
            help="Generate summary.csv after run completes.",
        ),
    ] = True,
    outliers: Annotated[
        str,
        typer.Option(
            "--outliers",
            help="Outlier filtering for summary: 'none' or 'iqr'.",
        ),
    ] = "iqr",
    do_report: Annotated[
        bool,
        typer.Option(
            "--report/--no-report",
            help="Generate report after run completes.",
        ),
    ] = True,
    report_format: Annotated[
        str,
        typer.Option(
            "--report-format",
            help="Report format: 'md' or 'html'.",
        ),
    ] = "md",
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots",
            help="Include distribution plots in report. Requires matplotlib for images.",
        ),
    ] = False,
    perf_stat: Annotated[
        bool,
        typer.Option(
            "--perf-stat/--no-perf-stat",
            help="Collect Linux perf stat counters (Linux only).",
        ),
    ] = False,
    perf_events: Annotated[
        str,
        typer.Option(
            "--perf-events",
            help="Comma-separated perf events to collect.",
        ),
    ] = "cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults",
    perf_strict: Annotated[
        bool,
        typer.Option(
            "--perf-strict/--no-perf-strict",
            help="Fail run if perf stat collection fails (otherwise degrades gracefully).",
        ),
    ] = False,
) -> None:
    """Run benchmarks using the specified configuration."""
    import random
    import secrets

    from stoatix.config import load_config
    from stoatix.plan import expand_suite
    from stoatix.results import (
        generate_suite_id,
        write_cases,
        write_resolved_config,
        write_session_metadata,
    )
    from stoatix.sysinfo import get_git_info, get_system_info

    # Validate outliers option early
    if outliers not in ("none", "iqr"):
        typer.echo(
            f"Error: Invalid --outliers value '{outliers}'. Must be 'none' or 'iqr'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate report format option early
    if report_format not in ("md", "html"):
        typer.echo(
            f"Error: Invalid --report-format value '{report_format}'. Must be 'md' or 'html'.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        # Always validate + expand config first
        config = load_config(config_path)
        cases = expand_suite(config)

        # Count cases per benchmark
        cases_per_bench: dict[str, int] = {}
        for case in cases:
            cases_per_bench[case.bench_name] = cases_per_bench.get(case.bench_name, 0) + 1

        if dry_run:
            # Print summary
            typer.echo(f"Configuration: {config_path}")
            typer.echo(f"Benchmarks: {len(config.benchmarks)}")
            typer.echo(f"Total cases: {len(cases)}")
            typer.echo()
            for bench in config.benchmarks:
                count = cases_per_bench.get(bench.name, 0)
                typer.echo(f"  {bench.name}: {count} case(s)")
            typer.echo()

            # Compute config hash
            import hashlib
            with open(config_path, "rb") as f:
                config_hash = hashlib.sha256(f.read()).hexdigest()

            # Handle shuffle seed for metadata
            actual_seed: int | None = None
            if shuffle:
                if seed is None:
                    actual_seed = secrets.randbits(32)
                else:
                    actual_seed = seed
                rng = random.Random(actual_seed)
                rng.shuffle(cases)
                typer.echo(f"Shuffle enabled with seed: {actual_seed}")

            suite_id = generate_suite_id()
            out.mkdir(parents=True, exist_ok=True)

            # Write metadata files
            write_session_metadata(
                out,
                suite_id=suite_id,
                config_path=str(config_path),
                config_hash=config_hash,
                benchmark_count=len(config.benchmarks),
                case_count=len(cases),
                shuffle_enabled=shuffle,
                seed=actual_seed,
                system_info=get_system_info(),
                git_info=get_git_info(),
            )
            write_resolved_config(out, config.to_resolved_dict())
            write_cases(out, cases, suite_id=suite_id)

            typer.echo(f"Dry run complete. Metadata written to: {out}")
            typer.echo("  - session.json")
            typer.echo("  - config.resolved.yml")
            typer.echo("  - cases.json")
            typer.echo("No benchmarks were executed (--dry-run mode).")
        else:
            # Full run
            from stoatix.runner import run_suite
            run_suite(
                config_path,
                out,
                shuffle=shuffle,
                seed=seed,
                perf_stat=perf_stat,
                perf_events=perf_events,
                perf_strict=perf_strict,
            )

            # Auto-summarize if enabled and results exist
            results_path = out / "results.jsonl"
            if do_summarize and results_path.exists():
                from stoatix.summarize import summarize_to_csv

                summary_path = out / "summary.csv"
                summaries = summarize_to_csv(
                    results_path,
                    summary_path,
                    outlier_method=outliers,  # type: ignore[arg-type]
                )

                # Print summary stats
                n_cases = len(summaries)
                total_ok = sum(s["n_ok"] for s in summaries)
                total_failed = sum(s["n_failed"] for s in summaries)

                typer.echo()
                typer.echo(f"Summary: {n_cases} case(s), {total_ok} OK, {total_failed} failed")
                typer.echo(f"  Written to: {summary_path}")

            # Auto-generate report if enabled and results exist
            if do_report and results_path.exists():
                from stoatix.report import generate_report, _HAS_MATPLOTLIB

                # Determine report output path based on format
                if report_format == "md":
                    report_out = out / "report.md"
                else:
                    report_out = out / "report.html"

                # Generate markdown report first
                md_report_path = generate_report(
                    results_path,
                    out_path=report_out if report_format == "md" else None,
                    outliers=outliers,  # type: ignore[arg-type]
                    plots=plots,
                )

                if report_format == "html":
                    # Convert to HTML
                    md_content = md_report_path.read_text(encoding="utf-8")
                    html_content = _markdown_to_html(md_content)
                    report_out.parent.mkdir(parents=True, exist_ok=True)
                    report_out.write_text(html_content, encoding="utf-8")
                    # Remove temporary md file if we generated it
                    if md_report_path != report_out:
                        md_report_path.unlink(missing_ok=True)
                    report_path = report_out
                else:
                    report_path = md_report_path

                typer.echo(f"Report: {report_path}")

                # Show plots info
                if plots:
                    if _HAS_MATPLOTLIB:
                        plots_dir = report_path.parent / "plots"
                        if plots_dir.exists():
                            num_plots = len(list(plots_dir.glob("*.png")))
                            typer.echo(f"  Boxplot images: {num_plots} in {plots_dir}")
                    else:
                        typer.echo("  Note: matplotlib not installed; only text histograms.")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def summarize(
    results_jsonl: Annotated[
        Path,
        typer.Argument(
            help="Path to results.jsonl file from a benchmark run.",
            exists=True,
            readable=True,
        ),
    ],
    out: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="Output CSV file path. Defaults to 'summary.csv' in same directory as results.",
        ),
    ] = None,
    outliers: Annotated[
        str,
        typer.Option(
            "--outliers",
            help="Outlier filtering method: 'none' (keep all) or 'iqr' (IQR-based filtering).",
        ),
    ] = "iqr",
) -> None:
    """Summarize benchmark results and write statistics to CSV."""
    from stoatix.summarize import summarize_to_csv

    # Validate outliers option
    if outliers not in ("none", "iqr"):
        typer.echo(
            f"Error: Invalid --outliers value '{outliers}'. Must be 'none' or 'iqr'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Determine output path
    if out is None:
        out = results_jsonl.parent / "summary.csv"

    try:
        summaries = summarize_to_csv(
            results_jsonl,
            out,
            outlier_method=outliers,  # type: ignore[arg-type]
        )

        # Compute totals
        n_cases = len(summaries)
        total_ok = sum(s["n_ok"] for s in summaries)
        total_failed = sum(s["n_failed"] for s in summaries)

        # Print summary
        typer.echo(f"Summarized {n_cases} case(s)")
        typer.echo(f"  Total OK iterations: {total_ok}")
        typer.echo(f"  Total failed iterations: {total_failed}")
        typer.echo(f"  Outlier filtering: {outliers}")
        typer.echo(f"CSV written to: {out}")

    except FileNotFoundError:
        typer.echo(
            f"Error: Results file not found: {results_jsonl}",
            err=True,
        )
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        typer.echo(
            f"Error: Failed to parse results file: {e.msg} at line {e.lineno}",
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def report(
    results_jsonl: Annotated[
        Path,
        typer.Argument(
            help="Path to results.jsonl file from a benchmark run.",
            exists=True,
            readable=True,
        ),
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'md' (Markdown) or 'html'.",
        ),
    ] = "md",
    out: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="Output file path. Defaults to 'report.md' or 'report.html' next to results.",
        ),
    ] = None,
    outliers: Annotated[
        str,
        typer.Option(
            "--outliers",
            help="Outlier filtering method: 'none' (keep all) or 'iqr' (IQR-based filtering).",
        ),
    ] = "iqr",
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots",
            help="Include distribution plots. Requires matplotlib for boxplot images.",
        ),
    ] = False,
) -> None:
    """Generate a report from benchmark results."""
    from stoatix.report import generate_report

    # Validate format option
    if format not in ("md", "html"):
        typer.echo(
            f"Error: Invalid --format value '{format}'. Must be 'md' or 'html'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate outliers option
    if outliers not in ("none", "iqr"):
        typer.echo(
            f"Error: Invalid --outliers value '{outliers}'. Must be 'none' or 'iqr'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Determine default output path based on format
    if out is None:
        if format == "md":
            out = results_jsonl.parent / "report.md"
        else:
            out = results_jsonl.parent / "report.html"

    try:
        # Generate markdown report first
        md_report_path = generate_report(
            results_jsonl,
            out_path=out if format == "md" else None,
            outliers=outliers,  # type: ignore[arg-type]
            plots=plots,
        )

        if format == "html":
            # Read markdown content and convert to HTML
            md_content = md_report_path.read_text(encoding="utf-8")
            html_content = _markdown_to_html(md_content)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html_content, encoding="utf-8")
            # Remove temporary md file if we generated it
            if md_report_path != out:
                md_report_path.unlink(missing_ok=True)
            report_path = out
        else:
            report_path = md_report_path

        typer.echo(f"Report written to: {report_path}")

        # Show notice about plots
        from stoatix.report import _HAS_MATPLOTLIB
        if plots:
            if _HAS_MATPLOTLIB:
                plots_dir = report_path.parent / "plots"
                if plots_dir.exists():
                    num_plots = len(list(plots_dir.glob("*.png")))
                    typer.echo(f"  Boxplot images: {num_plots} in {plots_dir}")
            else:
                typer.echo("  Note: matplotlib not installed; only text histograms included.")
                typer.echo("  Install with: uv add stoatix[viz]")

    except FileNotFoundError:
        typer.echo(
            f"Error: Results file not found: {results_jsonl}",
            err=True,
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def compare(
    main_jsonl: Annotated[
        Path,
        typer.Argument(
            help="Path to baseline results.jsonl (e.g., main branch).",
            exists=True,
            readable=True,
        ),
    ],
    pr_jsonl: Annotated[
        Path,
        typer.Argument(
            help="Path to comparison results.jsonl (e.g., PR branch).",
            exists=True,
            readable=True,
        ),
    ],
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Percentage threshold for classifying regressions/improvements.",
        ),
    ] = 0.05,
    outliers: Annotated[
        str,
        typer.Option(
            "--outliers",
            help="Outlier filtering method: 'iqr' or 'none'.",
        ),
    ] = "iqr",
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help="Metric to compare: 'median_s', 'mean_s', or 'p95_s'.",
        ),
    ] = "median_s",
    json_out: Annotated[
        Optional[Path],
        typer.Option(
            "--json-out",
            help="Output path for compare.json. Defaults to sibling of pr_jsonl.",
        ),
    ] = None,
    md_out: Annotated[
        Optional[Path],
        typer.Option(
            "--md-out",
            help="Output path for markdown report. If not set, only stdout.",
        ),
    ] = None,
    sort: Annotated[
        str,
        typer.Option(
            "--sort",
            help="Sort mode: 'stable' (by name) or 'priority' (regressed first).",
        ),
    ] = "priority",
    top: Annotated[
        int,
        typer.Option(
            "--top",
            help="Limit table rows in markdown output. 0 for unlimited.",
        ),
    ] = 50,
    noise_cv: Annotated[
        float,
        typer.Option(
            "--noise-cv",
            help="CV threshold for flagging noisy cases.",
        ),
    ] = 0.05,
    noise_p95_ratio: Annotated[
        float,
        typer.Option(
            "--noise-p95-ratio",
            help="P95/median ratio threshold for flagging noisy cases.",
        ),
    ] = 1.10,
    min_ok: Annotated[
        int,
        typer.Option(
            "--min-ok",
            help="Minimum OK iterations required.",
        ),
    ] = 3,
) -> None:
    """Compare two benchmark result files for regression detection."""
    from stoatix.compare import (
        compare_runs,
        render_compare_markdown,
        write_compare_json,
        write_compare_md,
    )

    # Validate outliers option
    if outliers not in ("iqr", "none"):
        typer.echo(
            f"Error: Invalid --outliers value '{outliers}'. Must be 'iqr' or 'none'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate metric option
    valid_metrics = ("median_s", "mean_s", "p95_s")
    if metric not in valid_metrics:
        typer.echo(
            f"Error: Invalid --metric value '{metric}'. Must be one of: {', '.join(valid_metrics)}.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate sort option
    if sort not in ("stable", "priority"):
        typer.echo(
            f"Error: Invalid --sort value '{sort}'. Must be 'stable' or 'priority'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate threshold
    if threshold < 0:
        typer.echo(
            "Error: --threshold must be non-negative.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate noise_cv
    if noise_cv < 0:
        typer.echo(
            "Error: --noise-cv must be non-negative.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate noise_p95_ratio
    if noise_p95_ratio < 1.0:
        typer.echo(
            "Error: --noise-p95-ratio must be >= 1.0.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate min_ok
    if min_ok < 1:
        typer.echo(
            "Error: --min-ok must be at least 1.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Determine default json_out path
    if json_out is None:
        json_out = pr_jsonl.parent / "compare.json"

    try:
        # Run comparison
        compare_result = compare_runs(
            main_results=main_jsonl,
            pr_results=pr_jsonl,
            threshold=threshold,
            outliers=outliers,  # type: ignore[arg-type]
            metric=metric,
            noise_cv_threshold=noise_cv,
            noise_p95_ratio_threshold=noise_p95_ratio,
            min_ok=min_ok,
        )

        # Render markdown
        md_content = render_compare_markdown(
            compare_result,
            top_n=top,
            sort_mode=sort,  # type: ignore[arg-type]
        )

        # Print to stdout
        typer.echo(md_content)

        # Always write JSON
        write_compare_json(json_out, compare_result)
        typer.echo(f"JSON written to: {json_out}", err=True)

        # Optionally write markdown
        if md_out is not None:
            write_compare_md(md_out, md_content)
            typer.echo(f"Markdown written to: {md_out}", err=True)

        # Exit with status based on regressions
        counts = compare_result["counts"]
        if counts["regressed"] > 0:
            typer.echo(
                f"\n⚠️  {counts['regressed']} regression(s) detected.",
                err=True,
            )

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        typer.echo(
            f"Error: Failed to parse results file: {e.msg} at line {e.lineno}",
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def _markdown_to_html(md_content: str) -> str:
    """Convert markdown content to a simple standalone HTML document.

    Uses basic regex-based conversion for tables and formatting.
    No external dependencies required.
    """
    import html
    import re

    lines = md_content.split("\n")
    html_lines: list[str] = []

    # HTML header with minimal styling
    html_lines.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stoatix Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3 { color: #2c3e50; margin-top: 1.5em; }
        h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 0.9em;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f5f5f5; }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85em;
        }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        ul { padding-left: 20px; }
        img { max-width: 100%; height: auto; margin: 1em 0; }
        em { font-style: italic; color: #7f8c8d; }
        hr { border: none; border-top: 1px solid #eee; margin: 2em 0; }
    </style>
</head>
<body>
""")

    in_table = False
    in_code_block = False
    table_rows: list[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if table_rows:
            html_lines.append("<table>")
            for i, row in enumerate(table_rows):
                cells = [c.strip() for c in row.strip("|").split("|")]
                tag = "th" if i == 0 else "td"
                html_lines.append(f"  <tr>{''.join(f'<{tag}>{html.escape(c)}</{tag}>' for c in cells)}</tr>")
            html_lines.append("</table>")
            table_rows = []
        in_table = False

    for line in lines:
        # Code block
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre>")
                in_code_block = False
            else:
                flush_table()
                html_lines.append("<pre>")
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        # Table row
        if "|" in line and not line.startswith("```"):
            # Skip separator rows
            if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
                continue
            if not in_table:
                flush_table()
                in_table = True
            table_rows.append(line)
            continue
        else:
            flush_table()

        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        # Horizontal rule
        elif line.startswith("---"):
            html_lines.append("<hr>")
        # List items
        elif line.startswith("- "):
            # Convert markdown links in list items
            content = line[2:]
            content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', content)
            html_lines.append(f"<ul><li>{content}</li></ul>")
        # Image
        elif re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line):
            match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line)
            if match:
                alt, src = match.groups()
                html_lines.append(f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">')
        # Bold text line
        elif line.startswith("**") and ":**" in line:
            # Key-value style like **Git:** value
            content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", line)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_lines.append(f"<p>{content}</p>")
        # Italic/emphasis
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            html_lines.append(f"<p><em>{html.escape(line.strip('*'))}</em></p>")
        # Empty line
        elif not line.strip():
            html_lines.append("")
        # Regular paragraph
        else:
            content = html.escape(line)
            # Convert inline code
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            # Convert links
            content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', content)
            # Convert bold
            content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
            if content.strip():
                html_lines.append(f"<p>{content}</p>")

    flush_table()

    html_lines.append("""
</body>
</html>""")

    return "\n".join(html_lines)


if __name__ == "__main__":
    app()
