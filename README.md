<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jdbadger/indexter/main/assets/indexter-light.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/jdbadger/indexter/main/assets/indexter-dark.svg">
    <img src="https://raw.githubusercontent.com/jdbadger/indexter/main/assets/indexter.png" alt="Indexter Logo">
  </picture>
</div>

<p align="center">
  <strong>Semantic Code Context For Your LLM</strong>
</p>

Indexter indexes your local git repositories, parses them semantically using tree-sitter, and provides a hybrid search interface for AI agents via the Model Context Protocol (MCP).

## Table of Contents

- [Features](#features)
- [Hybrid Search](#hybrid-search)
- [Supported Languages](#supported-languages)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [MCP Tools](#mcp-tools)
  - [Client Configuration](#client-configuration)
- [Contributing](#contributing)

## Features

- 🌳 **Semantic parsing** using tree-sitter for:
  - Python, JavaScript, TypeScript (including JSX/TSX), Rust
  - HTML, CSS, JSON, YAML, TOML, Markdown
  - Generic chunking fallback for other file types
- 📁 **Respects .gitignore** and configurable ignore patterns
- 🔄 **Incremental updates** sync changed files via document-level content hash comparison
- 🔍 **Hybrid search** combining dense semantic vectors and sparse keyword vectors with reciprocal rank fusion (RRF)
- ⚡ **Powered by Qdrant** vector database with automatic embedding generation via FastEmbed
- 🤖 **MCP server** for seamless AI agent integration via FastMCP
- 📦 **Multi-repo support** with separate collections per repository
- ⚙️ **XDG-compliant** configuration and data storage
- 👁️ **File watching** with automatic re-indexing on source changes (optional, with WSL polling fallback)

## Hybrid Search

Indexter uses **hybrid search** to combine the strengths of both semantic and keyword-based retrieval:

- **Dense Vectors**: Semantic embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`) capture the meaning and context of code, enabling natural language queries like "authentication handler" to find relevant code even without exact keyword matches.

- **Sparse Vectors**: BM25 keyword embeddings (default: `Qdrant/bm25`) provide traditional keyword-based search, ensuring exact matches for function names, variable names, and technical terms.

- **Reciprocal Rank Fusion (RRF)**: Results from both dense and sparse searches are combined and re-ranked using RRF, which:
  - Merges rankings from multiple retrieval methods
  - Reduces the impact of outliers from any single method
  - Provides more robust and relevant results than either approach alone

This hybrid approach ensures you get the best of both worlds: semantic understanding for conceptual queries and precision matching for specific identifiers.

## Supported Languages

Indexter uses tree-sitter for semantic parsing. Each parser extracts meaningful code units **along with their documentation** (docstrings, JSDoc, TSDoc, Rust doc comments, etc.):

| Language | Extensions | Semantic Units Extracted |
|----------|------------|-------------------------|
| Python | `.py` | Functions (sync/async), classes, decorated definitions, module-level constants + docstrings |
| JavaScript | `.js`, `.jsx` | Function declarations, generators, arrow functions, classes, methods + JSDoc comments |
| TypeScript | `.ts`, `.tsx` | Functions, generators, arrow functions, classes, interfaces, type aliases + TSDoc comments |
| Rust | `.rs` | Functions (sync/async/unsafe), structs, enums, traits, impl blocks + doc comments (`///`, `//!`) |
| HTML | `.html` | Semantic elements: tables, lists, headers (`<h1>`–`<h6>`) |
| CSS | `.css` | Rule sets, media queries, keyframes, imports, at-rules |
| JSON | `.json` | Objects, arrays |
| YAML | `.yaml`, `.yml` | Block mappings, block sequences |
| TOML | `.toml` | Tables, array tables, top-level pairs |
| Markdown | `.md`, `.mkd`, `.markdown` | ATX headings with section content |
| *Fallback* | `*` | Fixed-size overlapping chunks (for unsupported file types) |

## Prerequisites

- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/)
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant vector database)

## Installation

### Using uv

```bash
uv tool install "indexter"
```

### Using pipx

```bash
pipx install "indexter"
```

### From source

```bash
git clone https://github.com/jdbadger/indexter.git
cd indexter
uv sync
```

## Quickstart

Indexter is used primarily through MCP — your AI agent connects to the `indexter-mcp` server and interacts via tools.

**1. Start the Qdrant vector store** (one-time setup):

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ~/.local/share/indexter/qdrant:/qdrant/storage \
  qdrant/qdrant:latest
```

**2. Configure your MCP client** (see [Client Configuration](#client-configuration) for Claude Desktop, VS Code, and Cursor):

```json
{
  "mcpServers": {
    "indexter": {
      "command": "indexter-mcp"
    }
  }
}
```

**3. Use the tools through your AI agent:**

- `init_repo` — Register a git repository
- `index_repo` — Index (or re-index) a repository
- `search_repo` — Semantic search over indexed code
- `list_repos` — List all registered repositories
- `remove_repo` — Remove a repository and its data

## Configuration

Indexter uses a hierarchical, TOML-based configuration system with [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/latest/) compliance. Settings resolve through four levels — hard-coded defaults, global TOML, per-repo TOML, and environment variables.

| Priority | Source | Scope |
|----------|--------|-------|
| 1 (lowest) | Hard-coded defaults | All repos |
| 2 | `~/.config/indexter/indexter.toml` | All repos |
| 3 | `<repo>/indexter.toml` or `pyproject.toml` | Single repo |
| 4 (highest) | Environment variables (`INDEXTER_*`) | Runtime |

### Global Configuration

The global config file lives at `~/.config/indexter/indexter.toml` (or `$XDG_CONFIG_HOME/indexter/indexter.toml`) and is created with default values on first run.

```toml
# ~/.config/indexter/indexter.toml

# File patterns to exclude from indexing (gitignore-style syntax)
# These are in addition to patterns from .gitignore files
ignore_patterns = [
    ".git/",
    "__pycache__/",
    "*.pyc",
    ".DS_Store",
    "node_modules/",
    ".venv/",
    "*.lock",
]

max_file_size = 1048576  # Maximum file size in bytes (default: 1 MB)
max_files = 1000         # Maximum files per repository
top_k = 10               # Search results per query
upsert_batch_size = 32   # Documents per vector store batch

[store]
mode = "server"                                            # "server" or "memory"
image = "qdrant/qdrant:latest"                             # Docker image for Qdrant
host = "localhost"                                         # Qdrant server host
port = 6333                                                # Qdrant HTTP API port
grpc_port = 6334                                           # Qdrant gRPC port
prefer_grpc = false                                        # Prefer gRPC over HTTP
# api_key = "your-api-key"                                 # API key for authentication
embedding_model = "sentence-transformers/all-MiniLM-L6-v2" # Dense embedding model
sparse_embedding_model = "Qdrant/bm25"                     # Sparse embedding model
local_inference_batch_size = 32                             # Batch size for local inference

[mcp]
transport = "stdio"      # "stdio" or "http"
# host = "localhost"     # HTTP server host (only when transport = "http")
# port = 8765            # HTTP server port (only when transport = "http")

[watch]
enabled = false          # Enable background file watching
debounce_ms = 2000       # Delay (ms) after a change before re-indexing
poll_delay_ms = 5000     # Polling interval (ms) for environments without native FS events (e.g. WSL)
```

### Per-Repository Configuration

Override [top-level settings](#global-configuration) for a specific repository with an `indexter.toml` at the repo root, or a `[tool.indexter]` section in `pyproject.toml`. Only top-level fields (`ignore_patterns`, `max_file_size`, `max_files`, `top_k`, `upsert_batch_size`) can be overridden per-repo — `[store]`, `[mcp]`, and `[watch]` are global-only.

```toml
# <repo>/indexter.toml (or [tool.indexter] in pyproject.toml)

ignore_patterns = ["*.generated.*", "vendor/"]
max_files = 5000
```

Per-repo `ignore_patterns` are **merged** with global patterns (union), not replaced.

> For the full configuration reference — including all environment variables, directory structure, default ignore patterns, and programmatic access — see the [Configuration Guide](src/indexter/config/README.md).

## MCP Tools

Indexter exposes five tools via the [Model Context Protocol](https://modelcontextprotocol.io/), built on [FastMCP](https://gofastmcp.com). The server starts automatically when an MCP client connects (stdio mode).

| Tool | Description |
|------|-------------|
| `list_repos` | List all registered repositories with metadata and index staleness |
| `init_repo` | Register a new git repository (must contain a `.git` directory) |
| `index_repo` | Index or re-index a repository (incremental by default, `full=true` for rebuild) |
| `search_repo` | Hybrid semantic + keyword search with filters (language, node type, path, etc.) |
| `remove_repo` | Remove a repository and delete its indexed data |

> For full tool schemas, parameters, return types, and lifecycle details, see the [MCP Server Guide](src/indexter/mcp/README.md).

### Client Configuration

#### Claude Desktop & Claude Code

Add to your `claude_desktop_config.json` (located at `~/Library/Application Support/Claude/` on macOS or `%APPDATA%\Claude\` on Windows):

```json
{
  "mcpServers": {
    "indexter": {
      "command": "indexter-mcp"
    }
  }
}
```

If installed with uv:

```json
{
  "mcpServers": {
    "indexter": {
      "command": "uv",
      "args": ["tool", "run", "indexter-mcp"]
    }
  }
}
```

#### VS Code

Add to your VS Code settings (`.vscode/settings.json` or user settings):

```json
{
  "github.copilot.chat.mcp.servers": {
    "indexter": {
      "command": "indexter-mcp"
    }
  }
}
```

If installed with uv:

```json
{
  "github.copilot.chat.mcp.servers": {
    "indexter": {
      "command": "uv",
      "args": ["tool", "run", "indexter-mcp"]
    }
  }
}
```

#### Cursor

Add to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "indexter": {
      "command": "indexter-mcp"
    }
  }
}
```

If installed with uv:

```json
{
  "mcpServers": {
    "indexter": {
      "command": "uv",
      "args": ["tool", "run", "indexter-mcp"]
    }
  }
}
```

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/indexter.git
cd indexter

# Install dependencies with dev and test groups
uv sync --group dev --group test

# Run tests
uv run --group test pytest

# Run tests against all supported python versions
uv run just test
```

### Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to automatically run code quality checks before commits. The following hooks are configured:

- **File validation**: Check JSON, TOML, and YAML syntax, prevent large files
- **Dependency locking**: Keep `uv.lock` synchronized with `pyproject.toml`
- **Code formatting**: Format code with [Ruff](https://docs.astral.sh/ruff/)
- **Linting**: Lint and auto-fix issues with Ruff
- **Testing**: Run tests with [pytest](https://pytest.org/) and [testmon](https://testmon.org/) for fast incremental testing
- **Type checking**: Verify type hints with [ty](https://docs.astral.sh/ty/)

#### Setup

First, install pre-commit if you haven't already:

```bash
uv tool install pre-commit
```

Then initialize pre-commit for your clone:

```bash
pre-commit install
pre-commit install-hooks
```

#### Usage

Pre-commit hooks will now run automatically on `git commit`. To run all hooks manually:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run all hooks on staged files only
pre-commit run

# Run a specific hook
pre-commit run ruff-format --all-files
```

## License

MIT License - See [LICENSE](LICENSE) for details.
