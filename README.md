<div align="center">
  <img src="./indexter.svg" alt="Indexter Logo" style="filter: brightness(0) invert(1);">
</div>

<p align="center">
  <strong>Semantic Code Context For Your LLM</strong>
</p>

Indexter indexes your local git repositories, parses them semantically using tree-sitter, and provides a vector search interface for AI agents via the Model Context Protocol (MCP).

## Table of Contents

- [Features](#features)
- [Supported Languages](#supported-languages)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Modular Installation](#modular-installation)
  - [Using pipx](#using-pipx)
  - [From source](#from-source)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
  - [Global Configuration](#global-configuration)
  - [Per-Repository Configuration](#per-repository-configuration)
- [CLI Usage](#cli-usage)
  - [Examples](#examples)
- [MCP Usage](#mcp-usage)
  - [Claude Desktop](#claude-desktop)
  - [VS Code](#vs-code)
  - [Cursor](#cursor)
- [Programmatic Usage](#programmatic-usage)
- [Contributing](#contributing)

## Features

- 🌳 **Semantic parsing** using tree-sitter for:
  - Python, JavaScript, TypeScript (including JSX/TSX), Rust
  - HTML, CSS, JSON, YAML, TOML, Markdown
  - Generic chunking fallback for other file types
- 📁 **Respects .gitignore** and configurable ignore patterns
- 🔄 **Incremental updates** sync changed files via content hash comparison
- 🔍 **Vector search** powered by Qdrant with fastembed
- ⌨️ **CLI** for indexing repositories, searching code and inspecting configuration from your terminal
- 🤖 **MCP server** for seamless AI agent integration via FastMCP
- 📦 **Multi-repo support** with separate collections per repository
- ⚙️ **XDG-compliant** configuration and data storage

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
- [uv](https://docs.astral.sh/uv/) (recommended) or [pipx](https://pipx.pypa.io/)

## Installation

### Using uv (recommended)

To install the full application (CLI + MCP server):

```bash
uv tool install "indexter[full]"
```

### Modular Installation

Indexter is modular. You can install only the components you need:

- **Full Application** (CLI + MCP): `uv tool install "indexter[full]"`
- **CLI Only**: `uv tool install indexter[cli]`
- **MCP Server Only**: `uv tool install indexter[mcp]`
- **Core Library Only**: `uv add indexter[core]` (preferred: explicit > implicit) or `uv add indexter` - Useful for programmatic usage or building custom integrations.

### Using pipx

```bash
pipx install "indexter[full]"
```

### From source

```bash
git clone https://github.com/jdbadger/indexter.git
cd indexter
uv sync --all-extras
```

## Quickstart

```bash
# Initialize a repository for indexing
indexter init /path/to/your/repo/root

# Index the repository
indexter index your-repo-name

# Search the indexed code
indexter search "function that handles authentication" your-repo-name

# Check status of all indexed repositories
indexter status
```

## Configuration

### Global Configuration

Indexter uses XDG-compliant paths for configuration and data storage:

| Type | Path |
|------|------|
| Config | `~/.config/indexter/config.toml` |
| Data | `~/.local/share/indexter/` |

The global config controls embedding model, file processing settings, vector store, and MCP server:

```bash
# Show current configuration
indexter config show

# Get config file path
indexter config path

# Edit config manually
$EDITOR $(indexter config path)
```

```toml
# ~/.config/indexter/config.toml

# Embedding model to use for generating vector embeddings
embedding_model = "BAAI/bge-small-en-v1.5"

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
    # etc...
]

# Maximum file size (in bytes) to process
max_file_size = 1048576  # 1 MB

# Maximum number of files to process in a repository
max_files = 1000

# Number of top similar documents to retrieve for queries
top_k = 10

# Number of documents to upsert in a single batch operation
upsert_batch_size = 100

[store]
# Vector Store connection mode: 'local', 'remote', or 'memory'
mode = "local"

# Remote mode settings (only used when mode = "remote"):
# host = "localhost"          # Hostname of the remote Vector Store server
# port = 6333                 # HTTP API port
# grpc_port = 6334            # gRPC port
# prefer_grpc = false         # Whether to prefer gRPC over HTTP
# api_key = ""                # API key for authentication

[mcp]
# MCP transport mode: 'stdio' or 'http'
transport = "stdio"

# HTTP mode settings (only used when transport = "http"):
# host = "localhost"          # Hostname for the MCP HTTP server
# port = 8765                 # Port for the MCP HTTP server
```

**Store Modes:**
- `local`: File-based Qdrant in `$XDG_DATA_HOME/indexter` (default, no server required)
- `memory`: In-RAM, ephemeral — useful for testing
- `remote`: External Qdrant server — configure `host`, `port`, `grpc_port`, `prefer_grpc`, and `api_key`

**MCP Transports:**
- `stdio`: Standard input/output streams (default for MCP server integrations)
- `http`: HTTP server mode — configure `host` and `port`

Settings can also be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEXTER_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `INDEXTER_MAX_FILE_SIZE` | `1048576` | Maximum file size in bytes |
| `INDEXTER_MAX_FILES` | `1000` | Maximum files per repository |
| `INDEXTER_TOP_K` | `10` | Number of search results |
| `INDEXTER_UPSERT_BATCH_SIZE` | `100` | Batch size for vector operations |
| `INDEXTER_STORE_MODE` | `local` | Storage mode: `local`, `memory`, or `remote` |
| `INDEXTER_STORE_HOST` | `localhost` | Remote Qdrant host |
| `INDEXTER_STORE_PORT` | `6333` | Remote Qdrant HTTP API port |
| `INDEXTER_STORE_GRPC_PORT` | `6334` | Remote Qdrant gRPC port |
| `INDEXTER_STORE_PREFER_GRPC` | `false` | Prefer gRPC over HTTP |
| `INDEXTER_STORE_API_KEY` | `None` | Remote Qdrant API key |
| `INDEXTER_MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` or `http` |
| `INDEXTER_MCP_HOST` | `localhost` | MCP HTTP server host |
| `INDEXTER_MCP_PORT` | `8765` | MCP HTTP server port |

### Per-Repository Configuration

Create an `indexter.toml` in your repository root, or add a `[tool.indexter]` section to `pyproject.toml`:

```toml
# indexter.toml (or [tool.indexter] in pyproject.toml)

# Embedding model to use for this repository
# embedding_model = "BAAI/bge-small-en-v1.5"

# Additional patterns to ignore (combined with .gitignore and global patterns)
ignore_patterns = [
    "*.generated.*",
    "vendor/",
]

# Maximum file size (in bytes) to process. Default: 1048576 (1 MB)
# max_file_size = 1048576

# Maximum number of files to process in this repository. Default: 1000
# max_files = 1000

# Number of top similar documents to retrieve for queries. Default: 10
# top_k = 10

# Number of documents to batch when upserting to vector store. Default: 100
# upsert_batch_size = 100
```

## CLI Usage

```
indexter - Enhanced codebase context for AI agents via RAG.

Commands:
  init <path>           Initialize a git repository for indexing
  index <name>          Sync a repository to the vector store
  search <query> <name> Search indexed nodes in a repository
  status                Show status of indexed repositories
  forget <name>         Remove a repository from indexter
  config                View Indexter global settings
    show                Show global settings with syntax highlighting
    path                Print path to the settings config file

Options:
  --verbose, -v         Enable verbose output
  --version             Show version
  --help                Show help
```

### Examples

```bash
# Initialize and index a repository
indexter init ~/projects/my-repo
indexter index my-repo

# Force full re-index (ignores incremental sync)
indexter index my-repo --full

# Search with result limit
indexter search "error handling" my-repo --limit 5

# Forget a repository (removes from indexter and deletes indexed data)
indexter forget my-repo
```

## MCP Usage

Indexter provides an MCP server for AI agent integration. The server exposes:

| Type | Name | Description |
|------|------|-------------|
| Tool | `list_repositories` | List all configured repositories with their indexing status |
| Tool | `search_repository` | Semantic search across indexed code with filtering options |
| Prompt | `search_workflow` | Guide for effectively searching code repositories |

### Claude Desktop & Claude Code

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

### VS Code

Add to your VS Code settings (`.vscode/settings.json` in your workspace or user settings):

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

### Cursor

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

## Programmatic Usage

For custom integrations, use the `Repo` class directly:

```python
from indexter import Repo

# Initialize a new repository
repo = Repo.init("/path/to/your/repo", name="my-repo")

# Index the repository
await repo.index()

# Search indexed code
results = await repo.search("authentication handler", top_k=5)

# Check indexing status
status = await repo.status()

# Retrieve an existing repository
repo = Repo.get("my-repo")

# List all configured repositories
all_repos = Repo.all()

# Remove a repository
Repo.forget("my-repo")
```

Key properties: `repo.name`, `repo.path`, `repo.collection_name`, `repo.settings`.

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/indexter.git
cd indexter

# Install dependencies with all extras and test dependencies
uv sync --all-extras --group test

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
- **Type checking**: Verify type hints with [ty](https://github.com/jdbadger/ty)

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
