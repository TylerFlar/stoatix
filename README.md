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
uv run -- stoatix validate examples/stoatix.yaml

# Run benchmarks
uv run -- stoatix run examples/stoatix.yaml --out out/
```

## Results

After running benchmarks, results are written to the output directory:

- **`out/session.json`** — Metadata about the run including suite ID, config hash, system info, and git info
- **`out/runs.jsonl`** — One JSON record per benchmark iteration with timing data

## Configuration

Create a YAML file with your benchmarks:

```yaml
benchmarks:
  - name: my-benchmark
    command: ["python", "-c", "print('hello')"]
    warmups: 1      # warmup iterations (not measured)
    runs: 5         # measured iterations
    timeout_s: 10.0 # optional timeout in seconds
```

The `command` must be a list of strings (exec form, no shell).

## CLI Reference

```bash
# Show help
uv run -- stoatix --help

# Show version
uv run -- stoatix --version

# Validate config without running
uv run -- stoatix validate <config.yaml>

# Run benchmarks (verbose mode)
uv run -- stoatix --verbose run <config.yaml> --out <output_dir>
```

## License

MIT
