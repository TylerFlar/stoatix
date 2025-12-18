# Contributing to Stoatix

Thank you for your interest in contributing to Stoatix!

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/TylerFlar/stoatix.git
   cd stoatix
   ```

2. Install dependencies (including dev dependencies):

   ```bash
   uv sync
   ```

## Development Commands

### Run Tests

```bash
uv run -- pytest
```

### Lint Code

```bash
uv run -- ruff check .
```

### Format Code

```bash
uv run -- ruff format .
```

## Code Style

- Use type hints for all function signatures
- Keep code readable and well-documented
- Run `ruff check .` and `ruff format .` before committing

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
