"""Stoatix CLI entry point."""

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
            run_suite(config_path, out, shuffle=shuffle, seed=seed)

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
