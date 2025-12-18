# Stoatix

A lightweight CLI tool for system benchmarking and diagnostics.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/TylerFlar/stoatix.git
cd stoatix

# Install dependencies
uv sync

# Validate a configuration file
uv run stoatix validate examples/stoatix.yaml

# Dry run (validate + write metadata without executing)
uv run stoatix run examples/stoatix.yaml --dry-run

# Run benchmarks
uv run stoatix run examples/stoatix.yaml --out out/

# Run with shuffle (randomized case order)
uv run stoatix run examples/stoatix.yaml --shuffle --seed 12345
```

## Results

After running benchmarks, results are written to the output directory:

- **`out/session.json`** — Metadata about the run including suite ID, config hash, system info, and git info
- **`out/config.resolved.yml`** — Resolved configuration with all defaults expanded
- **`out/cases.json`** — Expanded cases in execution order
- **`out/results.jsonl`** — One JSON record per benchmark attempt with timing data

## Configuration

Create a YAML file with your benchmarks:

```yaml
# Optional defaults applied to all benchmarks
defaults:
  warmups: 2
  runs: 5
  retries: 0
  timeout_s: 60.0
  env:
    MY_VAR: "value"

benchmarks:
  - name: my-benchmark
    command: ["python", "-c", "print('hello')"]
    warmups: 1      # warmup iterations (not measured)
    runs: 5         # measured iterations
    timeout_s: 10.0 # optional timeout in seconds

  # Matrix expansion example
  - name: parameterized-bench
    command: ["python", "-c", "print({n})"]
    matrix:
      n: [100, 1000, 10000]
```

The `command` must be a list of strings (exec form, no shell).

## CLI Reference

```bash
# Show help
uv run stoatix --help

# Show version
uv run stoatix --version

# Validate config without running
uv run stoatix validate <config.yaml>

# Run benchmarks
uv run stoatix run <config.yaml> [OPTIONS]

# Run command options:
#   --out, -o PATH      Output directory (default: out)
#   --shuffle           Shuffle case execution order
#   --no-shuffle        Preserve deterministic order (default)
#   --seed INTEGER      Random seed for shuffling
#   --dry-run           Write metadata files without executing benchmarks

# Examples:
uv run stoatix run config.yaml --out results/
uv run stoatix run config.yaml --shuffle --seed 42
uv run stoatix run config.yaml --dry-run
uv run stoatix --verbose run config.yaml
```

## License

MIT
