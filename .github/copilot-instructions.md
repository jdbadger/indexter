# Indexter - Copilot Instructions

## Project Summary

Indexter is a CLI tool and MCP (Model Context Protocol) server that indexes local git repositories using tree-sitter semantic parsing and provides hybrid search (semantic + keyword) via Qdrant vector database. It enables AI agents to efficiently search and understand codebases.

**Key Features**: Tree-sitter parsing for 10+ languages, hybrid search with RRF (Reciprocal Rank Fusion), incremental indexing, XDG-compliant config, multi-repo support.

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync --group dev` or `uv sync --group test` |
| Run tests | `uv run --group test pytest` |
| Run tests with coverage | `uv run --group test pytest --cov=indexter --cov-fail-under=95` |
| Lint check | `uv run --group dev ruff check src/indexter` |
| Lint fix | `uv run --group dev ruff check --fix src/indexter` |
| Format check | `uv run --group dev ruff format --check src/indexter` |
| Format fix | `uv run --group dev ruff format src/indexter` |
| Type check | `uv run --group dev ty check src/indexter` |
| Build package | `uv build` |

## Tech Stack & Versions

- **Language**: Python 3.11, 3.12, 3.13 (target: 3.13)
- **Package Manager**: uv (0.9+)
- **Build Backend**: uv_build
- **Linter/Formatter**: Ruff 0.14+
- **Type Checker**: ty
- **Testing**: pytest, pytest-asyncio, pytest-cov, pytest-testmon
- **Vector Store**: Qdrant (via qdrant-client with FastEmbed)
- **Parsing**: tree-sitter with language pack
- **MCP**: FastMCP

## Project Layout

```
indexter/
├── pyproject.toml         # Project config, deps, tool settings
├── justfile               # Task runner for releases and multi-version tests
├── .pre-commit-config.yaml # Pre-commit hooks config
├── .python-version        # Default Python version (3.13)
├── src/indexter/          # Main source code
│   ├── __init__.py        # Exports: Repo, __version__
│   ├── config.py          # Configuration system (XDG-compliant, hierarchical)
│   ├── models.py          # Core Repo model with async index/search/status
│   ├── exceptions.py      # Custom exceptions: RepoNotFoundError, RepoExistsError
│   ├── cli/               # CLI module (Typer-based)
│   │   ├── cli.py         # CLI commands
│   │   └── tests/         # CLI tests
│   ├── mcp/               # MCP server module (FastMCP-based)
│   │   ├── server.py      # MCP server definition and tools
│   │   ├── tools.py       # Tool implementations
│   │   └── tests/         # MCP tests
│   ├── parser/            # Tree-sitter parsing
│   │   ├── parser.py      # Parser factory
│   │   ├── models.py      # NodeMetadata model
│   │   └── parsers/       # Language-specific parsers
│   ├── store/             # Vector store (Qdrant)
│   │   ├── store.py       # VectorStore class
│   │   └── tests/         # Store tests
│   └── walker/            # File system walker
│       ├── walker.py      # Walker class
│       └── tests/         # Walker tests
└── .github/
    ├── workflows/
    │   ├── ci.yml         # CI: tests on Python 3.11/3.12/3.13, coverage ≥95%
    │   └── publish.yml    # Publish to PyPI on release
    └── instructions/      # Copilot context instructions
```

## Build & Development Workflow

### 1. Environment Setup (ALWAYS do first)

```bash
# Install all dependencies (dev includes test tools)
uv sync --group test
```

**Important**: Always use `uv run --group test` or `uv run --group dev` prefix for commands.

### 2. Before Making Changes

```bash
# Lint the file you're about to modify
uv run --group dev ruff check src/indexter/path/to/file.py

# Type check the file
uv run --group dev ty check src/indexter/path/to/file.py
```

### 3. After Making Changes

```bash
# Format code
uv run --group dev ruff format src/indexter

# Lint and auto-fix
uv run --group dev ruff check --fix src/indexter

# Type check
uv run --group dev ty check src/indexter

# Run tests with coverage (CI requires ≥95%)
uv run --group test pytest --cov=indexter --cov-fail-under=95
```

### 4. Running Specific Tests

```bash
# Run a specific test file
uv run --group test pytest src/indexter/parser/tests/test_parser.py

# Run tests matching a pattern
uv run --group test pytest -k "test_search"

# Run with verbose output
uv run --group test pytest -v
```

## CI Requirements

The CI pipeline (`.github/workflows/ci.yml`) runs on every PR and push to main:

1. **Tests across Python 3.11, 3.12, 3.13**
2. **Coverage ≥95%** is enforced (`--cov-fail-under=95`)

Always run the full test suite with coverage before submitting changes.

## Pre-commit Hooks

The project uses pre-commit hooks that run on `git commit`:
- JSON/TOML/YAML validation
- uv lock sync
- Ruff format and lint
- pytest with testmon (incremental)
- ty type checking

To run manually: `pre-commit run --all-files`

## Key Code Patterns

### Async-First Design
Most operations are async. Use `async/await` consistently:
```python
async def example():
    results = await store.search(collection_name, query)
```

### Testing
Tests are co-located with modules in `tests/` subdirectories. Use pytest fixtures:
```python
@pytest.fixture
def sample_repo(tmp_path):
    ...
```
See [.github/instructions/python-tests.instructions.md](.github/instructions/python-tests.instructions.md) for detailed test writing guidelines (AAA pattern, fixtures, mocking, parameterized tests). This file is automatically applied when working with `**/*.py` files.

### Configuration
Settings are loaded hierarchically: defaults → global config → repo config → env vars.
Access via `from indexter.config import settings`.

## Entry Points

- **CLI**: `indexter` → `src/indexter/cli/cli.py:app`
- **MCP Server**: `indexter-mcp` → `src/indexter/mcp/server.py:server.run`
- **Programmatic**: `from indexter import Repo`

## Common Issues & Solutions

1. **Missing dependencies**: Always run `uv sync --group test` first
2. **Test failures on coverage**: The threshold is 95%. Check `--cov-report=term-missing` for gaps
3. **Type errors**: Use `uv run --group dev ty check` to identify issues
4. **Async test errors**: All async tests run with `asyncio_mode = "auto"` in pytest config

## Trust These Instructions

These instructions have been validated against the actual codebase. Only perform additional exploration if:
- The documented command fails
- You need information not covered here
- The codebase structure has significantly changed
