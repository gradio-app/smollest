# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- AVOID inline comments in code!!!

## Commands

### Development Setup
```bash
# Install with development dependencies
pip install -e .[dev]
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_compare.py
```

### Code Formatting and Linting
```bash
# Format and lint code (required before commits)
ruff check --fix --select I && ruff format
```

## Architecture Overview

smollest (Minimum Viable Language Model) provides drop-in wrappers for OpenAI and Anthropic clients that silently replay requests to smaller candidate LLMs and compare structured outputs.

### Key Modules

- `smollest/openai.py` — Drop-in `OpenAI` wrapper, intercepts `chat.completions.create()`
- `smollest/anthropic.py` — Drop-in `Anthropic` wrapper, intercepts `messages.create()`
- `smollest/candidates.py` — Runs candidate models (HuggingFace serverless + OpenAI-compatible local servers)
- `smollest/compare.py` — JSON field-by-field exact match comparison
- `smollest/results.py` — Logging to JSON file + console output

### Testing Strategy

Tests are located in the `tests/` directory. Always run tests before committing changes.

### Important Files for Common Tasks

- **Adding new features**: Modify relevant modules in `smollest/`
- **Adding dependencies**: Update `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`

## Issue Resolution Workflow

When given a GitHub issue to solve, follow this workflow:

1. **Create a new branch** named after the issue (e.g., `fix-issue-123` or descriptive name)
2. **Implement the solution** following the existing code patterns and conventions
3. **Run tests** to ensure nothing is broken: `pytest`
4. **Run linting/formatting**: `ruff check --fix --select I && ruff format`
5. That's it. Never use the `git` commit command after a task is finished.

### Git Commands for Issue Workflow
```bash
git checkout -b fix-issue-NUMBER
```

Always ensure tests pass and code is formatted before pushing.
