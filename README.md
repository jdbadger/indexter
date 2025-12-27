<div align="center">
  <img src="./indexter.svg" alt="Indexter Logo" style="filter: brightness(0) invert(1);">
</div>

<p align="center">
  <strong>Semantic Code Context For Your LLM</strong>
</p>

Indexter indexes your local git repositories, parses them semantically using tree-sitter, and provides a vector search interface for AI agents via the Model Context Protocol (MCP).

## Table of Contents

- [Features](#features)
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
- [Contributing](#contributing)

## Features

- 🌳 **Semantic parsing** using tree-sitter for:
  - Python, JavaScript, TypeScript (including JSX/TSX)
  - Rust, HTML, CSS
  - JSON, YAML, TOML, Markdown
  - Generic chunking fallback for other file types
- 📁 **Respects .gitignore** and configurable ignore patterns
- 🔄 **Incremental updates** sync changed files via content hash comparison
- 🔍 **Vector search** powered by Qdrant with fastembed
- ⌨️ **CLI** for config management, indexing repositories, and searching code from your terminal
- 🤖 **MCP server** for seamless AI agent integration via FastMCP
- 📦 **Multi-repo support** with separate collections per repository
- ⚙️ **XDG-compliant** configuration and data storage

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
git clone https://github.com/YOUR_ORG/indexter.git
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

Indexter uses XDG-compliant paths for configuration and data storage:

| Type | Path |
|------|------|
| Config | `~/.config/indexter/config.toml` |
| Data | `~/.local/share/indexter/` |
| Cache | `~/.cache/indexter/` |

### Global Configuration

The global config controls vector store, embedding model, and MCP server settings:

```bash
# Show current configuration
indexter config show

# Create/reset global config
indexter config init

# Edit config in $EDITOR
indexter config edit
```

```toml
# ~/.config/indexter/config.toml

# Default ignore patterns (applied to all repositories)
# These use gitignore-style patterns and are in addition to .gitignore
# Uncomment and modify to customize defaults for all repositories
# default_ignore_patterns = [
#     ".git/",
#     "__pycache__/",
#     "*.pyc",
#     "node_modules/",
#     ".venv/",
#     "*.lock",
# ]

[store]
# Connection mode: "local" (serverless), "memory" (testing), or "remote"
mode = "local"
# path = "~/.local/share/indexter/store"  # Custom storage path

# Remote mode settings (only used when mode = "remote")
# host = "localhost"
# port = 6333
# api_key = ""

[embedding]
model_name = "BAAI/bge-small-en-v1.5"

[mcp]
host = "localhost"
port = 8765
default_top_k = 10
```

Settings can also be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEXTER_STORE_MODE` | `local` | Storage mode: `local`, `memory`, or `remote` |
| `INDEXTER_STORE_HOST` | `localhost` | Remote Qdrant host |
| `INDEXTER_STORE_PORT` | `6333` | Remote Qdrant port |
| `INDEXTER_MCP_PORT` | `8765` | MCP server port |

### Per-Repository Configuration

Create an `indexter.toml` in your repository root, or add a `[tool.indexter]` section to `pyproject.toml`:

```toml
# indexter.toml (or [tool.indexter] in pyproject.toml)

# Override the auto-generated collection name
# collection = "my-custom-collection"

# Additional patterns to ignore (in addition to .gitignore)
ignore_patterns = [
    "*.generated.*",
    "vendor/",
]

# Maximum file size to process (in bytes). Default: 10MB
max_file_size = 10485760

# Maximum number of files to sync in a single operation. Default: 500
# Useful for large repositories to prevent overwhelming the system
max_sync_files = 500

# Number of nodes to batch when upserting to vector store. Default: 100
# Tune this for optimal performance based on your system's memory and network
upsert_batch_size = 100
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
  config                Manage global configuration

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
| Tool | `sync` | Sync a repository's vector index with local file state |
| Tool | `search` | Semantic search across indexed code with filtering options |
| Resource | `repos://list` | List all configured repositories |
| Resource | `repo://{name}/status` | Get indexing status of a repository |
| Prompt | `search_workflow` | Guide for effectively searching code repositories |

### Claude Desktop

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

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/indexter.git
cd indexter

# Install dependencies
uv sync

# Run tests
uv run --group dev pytest
```

## License

MIT License - See [LICENSE](LICENSE) for details.
