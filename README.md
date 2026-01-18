<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jdbadger/indexter/main/indexter-light.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/jdbadger/indexter/main/indexter-dark.svg">
    <img src="https://raw.githubusercontent.com/jdbadger/indexter/main/indexter.png" alt="Indexter Logo">
  </picture>
</div>

<p align="center">
  <strong>Semantic Code Context For Your LLM</strong>
</p>

Indexter indexes your local git repositories, parses them semantically using tree-sitter, and provides a hybrid search interface for AI agents via the Model Context Protocol (MCP).

## Table of Contents

- [Features](#features)
- [Supported Languages](#supported-languages)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using uv](#using-uv)
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
- 🔍 **Hybrid search** combining dense semantic vectors and sparse keyword vectors with reciprocal rank fusion (RRF)
- ⚡ **Powered by Qdrant** vector database with automatic embedding generation via FastEmbed
- ⌨️ **CLI** for indexing repositories, searching code and inspecting configuration from your terminal
- 🤖 **MCP server** for seamless AI agent integration via FastMCP
- 📦 **Multi-repo support** with separate collections per repository
- ⚙️ **XDG-compliant** configuration and data storage

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
# Initialize the Qdrant vector store (pulls Docker image and starts container)
indexter store init

# Initialize and index a repository (indexes automatically by default)
indexter init --path /path/to/your/repo/root

# Or initialize current directory
indexter init

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
| Config | `~/.config/indexter/indexter.toml` |
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
# ~/.config/indexter/indexter.toml

# Dense embedding model for semantic search (default: 384-dim sentence-transformers model)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Sparse embedding model for keyword-based search (BM25 algorithm)
sparse_embedding_model = "Qdrant/bm25"

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
# Vector Store connection mode: 'server' or 'memory'
mode = "server"

# Docker image for the Qdrant container (used by 'indexter store init')
image = "qdrant/qdrant:latest"

# Timeout in seconds for API operations (increase if experiencing timeouts)
timeout = 120

# Server mode settings (used when mode = "server"):
# host = "localhost"          # Hostname of the Qdrant server
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
- `server`: Docker-managed Qdrant container (default, managed via `indexter store` commands)
- `memory`: In-RAM, ephemeral — useful for testing

**MCP Transports:**
- `stdio`: Standard input/output streams (default for MCP server integrations)
- `http`: HTTP server mode — configure `host` and `port`

Settings can also be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEXTER_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model for semantic search |
| `INDEXTER_SPARSE_EMBEDDING_MODEL` | `Qdrant/bm25` | Sparse embedding model for keyword search |
| `INDEXTER_MAX_FILE_SIZE` | `1048576` | Maximum file size in bytes |
| `INDEXTER_MAX_FILES` | `1000` | Maximum files per repository |
| `INDEXTER_TOP_K` | `10` | Number of search results |
| `INDEXTER_UPSERT_BATCH_SIZE` | `100` | Batch size for vector operations |
| `INDEXTER_STORE_MODE` | `server` | Storage mode: `server` or `memory` |
| `INDEXTER_STORE_IMAGE` | `qdrant/qdrant:latest` | Docker image for Qdrant container |
| `INDEXTER_STORE_HOST` | `localhost` | Qdrant server host |
| `INDEXTER_STORE_PORT` | `6333` | Qdrant HTTP API port |
| `INDEXTER_STORE_GRPC_PORT` | `6334` | Qdrant gRPC port |
| `INDEXTER_STORE_PREFER_GRPC` | `false` | Prefer gRPC over HTTP |
| `INDEXTER_STORE_API_KEY` | `None` | Qdrant API key |
| `INDEXTER_STORE_TIMEOUT` | `120` | API operation timeout (seconds) |
| `INDEXTER_MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` or `http` |
| `INDEXTER_MCP_HOST` | `localhost` | MCP HTTP server host |
| `INDEXTER_MCP_PORT` | `8765` | MCP HTTP server port |

### Per-Repository Configuration

Create an `indexter.toml` in your repository root, or add a `[tool.indexter]` section to `pyproject.toml`:

```toml
# indexter.toml (or [tool.indexter] in pyproject.toml)

# Dense embedding model for semantic search
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Sparse embedding model for keyword-based search
# sparse_embedding_model = "Qdrant/bm25"

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

# Number of documents to batch when upserting to vector store. Default: 50
# upsert_batch_size = 100
```

## CLI Usage

```
indexter - Enhanced codebase context for AI agents via RAG.

Commands:
  init                  Initialize a git repository for indexing
    --path, -p          Path to the git repository (defaults to current directory)
    --no-index, -n      Skip indexing after initialization
  index <name>          Sync a repository to the vector store
  search <query> <name> Search indexed nodes in a repository
  status                Show status of indexed repositories
  forget <name>         Remove a repository from indexter
  config                View Indexter global settings
    show                Show global settings with syntax highlighting
    path                Print path to the settings config file
  store                 Manage the Qdrant vector store container
    init                Initialize the store (pull image and start container)
    start               Start the store container
    status              Show store container status
    stop                Stop the store container
    remove              Remove the store container
      --volumes, -v     Also remove data volumes

Options:
  --verbose, -v         Enable verbose output
  --version             Show version
  --help                Show help
```

### Store Management

Indexter uses Qdrant as its vector database, running in a Docker container. The `indexter store` commands manage this container lifecycle:

```bash
# Initialize the store (pulls image, creates and starts container)
indexter store init

# Check container status
indexter store status

# Stop the container (data is preserved)
indexter store stop

# Start a stopped container
indexter store start

# Remove the container (data is preserved in ~/.local/share/indexter/qdrant)
indexter store remove

# Remove the container and all stored data
indexter store remove --volumes
```

The store data is persisted in `~/.local/share/indexter/qdrant` via a bind mount, so your indexed data survives container restarts and removals (unless `--volumes` is used).

### Examples

```bash
# Initialize and index a repository
indexter init --path ~/projects/my-repo

# Initialize without indexing (index later)
indexter init --path ~/projects/my-repo --no-index
indexter index my-repo

# Initialize current directory
indexter init

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
| Tool | `get_repository` | Get metadata for a specific repository |
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

For custom integrations, use the `Repo` class with a `VectorStore` context manager:

```python
import asyncio
from pathlib import Path
from indexter import Repo
from indexter.store import VectorStore

async def main():
    async with VectorStore() as store:
        # Initialize a new repository (name derived from directory)
        repo = await Repo.init(Path("/path/to/your/repo"), store)

        # Index the repository
        result = await repo.index(store)
        print(f"Indexed {result.nodes_added} nodes")

        # Search indexed code
        results = await repo.search("authentication handler", store, limit=5)
        for r in results.results:
            print(f"{r.score:.3f}: {r.metadata['node_name']}")

        # Retrieve an existing repository with status metadata
        repo = await Repo.get_one("my-repo", store, with_metadata=True)
        print(f"Stale: {repo.metadata.is_stale}")

        # List all configured repositories
        all_repos = await Repo.get_all(store)

        # Remove a repository and its indexed data
        await Repo.remove_one("my-repo", store)

asyncio.run(main())
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
