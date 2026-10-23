# Contributing to windtunnel-shear

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/MothLabs-org/windtunnel-shear.git
cd windtunnel-shear

# Install in dev mode
make install
```

This installs the package in editable mode with all dev dependencies and sets up pre-commit hooks.

## Running Tests

```bash
make test          # Run all tests
make coverage      # Run with coverage (must be >= 90%)
make lint          # Lint check
make format        # Auto-format code
make typecheck     # Run mypy
```

## Code Style

- **Type hints everywhere.** All function signatures and return types.
- **`from __future__ import annotations`** in every file.
- **Google-style docstrings** on every public function.
- **ruff** for linting and formatting (configured in `pyproject.toml`).
- **No `Any` leakage** — use `Any` only in raw passthrough dicts.

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure `make lint`, `make typecheck`, and `make test` all pass
4. Update `CHANGELOG.md` under `[Unreleased]` if your change is user-facing
5. Open a PR with a clear description of the change

## Issue Labels

- `bug` — Something isn't working
- `feature` — New feature request
- `good-first-issue` — Good for newcomers
- `help-wanted` — Extra attention is needed

## Questions?

Open an issue or start a discussion on the repo.
