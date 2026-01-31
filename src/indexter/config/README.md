# Configuration

Indexter uses a hierarchical, TOML-based configuration system with [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/latest/) compliance. Settings resolve through four levels — hard-coded defaults, global TOML, per-repo TOML, and environment variables — so you can tune behavior globally or per-project without touching code.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Hierarchy](#configuration-hierarchy)
- [Global Configuration](#global-configuration)
  - [Top-Level Settings](#top-level-settings)
  - [\[store\] — Vector Store](#store--vector-store)
  - [\[mcp\] — MCP Server](#mcp--mcp-server)
  - [\[watch\] — File Watcher](#watch--file-watcher)
  - [Full Example](#full-example)
- [Per-Repository Configuration](#per-repository-configuration)
- [Environment Variables](#environment-variables)
  - [Common Recipes](#common-recipes)
- [Directory Structure](#directory-structure)
- [Default Ignore Patterns](#default-ignore-patterns)
- [Programmatic Access](#programmatic-access)

## Quick Start

Indexter works out of the box with sensible defaults. The global config file is created automatically on first run:

```toml
# ~/.config/indexter/indexter.toml

top_k = 20              # return more results per search
max_files = 5000        # index larger repos
```

Or override a single setting via environment variable:

```bash
INDEXTER_TOP_K=20 indexter serve
```

## Configuration Hierarchy

Settings are resolved in the following order. Each level overrides the previous:

| Priority | Source | Scope |
|----------|--------|-------|
| 1 (lowest) | Hard-coded defaults | All repos |
| 2 | `~/.config/indexter/indexter.toml` | All repos |
| 3 | `<repo>/indexter.toml` or `pyproject.toml` | Single repo |
| 4 (highest) | Environment variables (`INDEXTER_*`) | Runtime |

**Example** — how `top_k` resolves through all four levels:

1. Default: `top_k = 10`
2. Global TOML sets `top_k = 20` → **20**
3. Repo TOML sets `top_k = 5` → **5** (for this repo only)
4. `INDEXTER_TOP_K=50` → **50** (overrides everything at runtime)

## Global Configuration

The global config file lives at `~/.config/indexter/indexter.toml` (or `$XDG_CONFIG_HOME/indexter/indexter.toml`). It is created with default values on first run.

### Top-Level Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ignore_patterns` | `list[str]` | [See defaults](#default-ignore-patterns) | File/directory patterns to exclude from indexing |
| `max_file_size` | `int` | `1048576` (1 MB) | Maximum file size in bytes to process |
| `max_files` | `int` | `1000` | Maximum files to index per repository |
| `top_k` | `int` | `10` | Number of results returned per search query |
| `upsert_batch_size` | `int` | `32` | Documents per vector store batch operation |

### `[store]` — Vector Store

Connection settings for the Qdrant vector database.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"server"` \| `"memory"` | `"server"` | Connection mode. Use `"memory"` for testing |
| `image` | `str` | `"qdrant/qdrant:latest"` | Docker image for the Qdrant container |
| `host` | `str` | `"localhost"` | Qdrant server hostname |
| `port` | `int` | `6333` | Qdrant HTTP API port |
| `grpc_port` | `int` | `6334` | Qdrant gRPC port |
| `prefer_grpc` | `bool` | `false` | Prefer gRPC over HTTP for connections |
| `api_key` | `str` or unset | unset | API key for authenticated connections |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Dense embedding model |
| `sparse_embedding_model` | `str` | `"Qdrant/bm25"` | Sparse embedding model for keyword search |
| `local_inference_batch_size` | `int` | `32` | Batch size for local embedding inference |

> **Note**: `host`, `port`, `grpc_port`, `prefer_grpc`, and `api_key` only apply when `mode = "server"`.

### `[mcp]` — MCP Server

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transport` | `"stdio"` \| `"http"` | `"stdio"` | MCP transport mode |
| `host` | `str` | `"localhost"` | HTTP server hostname |
| `port` | `int` | `8765` | HTTP server port |

> **Note**: `host` and `port` only apply when `transport = "http"`.

### `[watch]` — File Watcher

Background file watching for automatic re-indexing when source files change.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable background file watching |
| `debounce_ms` | `int` | `2000` | Milliseconds to wait after a change before re-indexing |
| `poll_delay_ms` | `int` | `5000` | Polling interval (ms) for environments without native FS events (e.g. WSL) |

### Full Example

```toml
# ~/.config/indexter/indexter.toml

ignore_patterns = [
    ".git/",
    "__pycache__/",
    "*.pyc",
    "node_modules/",
    "*.lock",
]

# Maximum file size (in bytes) to process
max_file_size = 1048576

# Maximum number of files to process in a repository
max_files = 1000

# Number of top similar documents to retrieve for queries
top_k = 10

# Number of documents to upsert in a single batch operation
upsert_batch_size = 32

[store]
image = "qdrant/qdrant:latest"
mode = "server"
host = "localhost"
port = 6333
grpc_port = 6334
prefer_grpc = false
# api_key = "your-api-key"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
sparse_embedding_model = "Qdrant/bm25"
local_inference_batch_size = 32

[mcp]
transport = "stdio"
# host and port apply only when transport = "http"
# host = "localhost"
# port = 8765

[watch]
enabled = false
debounce_ms = 2000
poll_delay_ms = 5000
```

## Per-Repository Configuration

Override settings for a specific repository using either an `indexter.toml` file or a `[tool.indexter]` section in `pyproject.toml` at the repository root.

Only [top-level settings](#top-level-settings) can be overridden per-repo — `[store]`, `[mcp]`, and `[watch]` are global-only.

**Priority**: `indexter.toml` takes precedence over `pyproject.toml` if both exist.

**Ignore patterns**: Per-repo patterns are **merged** with global patterns (union), not replaced. This ensures global exclusions always apply.

### `indexter.toml`

```toml
# <repo>/indexter.toml

max_files = 5000
ignore_patterns = ["docs/build/", "*.generated.ts"]
```

### `pyproject.toml`

```toml
# <repo>/pyproject.toml

[tool.indexter]
max_files = 5000
ignore_patterns = ["docs/build/", "*.generated.ts"]
```

## Environment Variables

Environment variables override all file-based configuration. Use them for runtime overrides, CI, or containerized deployments.

### Top-Level (`INDEXTER_*`)

| Variable | Field | Example |
|----------|-------|---------|
| `INDEXTER_MAX_FILE_SIZE` | `max_file_size` | `2097152` |
| `INDEXTER_MAX_FILES` | `max_files` | `5000` |
| `INDEXTER_TOP_K` | `top_k` | `20` |
| `INDEXTER_UPSERT_BATCH_SIZE` | `upsert_batch_size` | `64` |

### Store (`INDEXTER_STORE_*`)

| Variable | Field | Example |
|----------|-------|---------|
| `INDEXTER_STORE_MODE` | `mode` | `memory` |
| `INDEXTER_STORE_IMAGE` | `image` | `qdrant/qdrant:v1.8.0` |
| `INDEXTER_STORE_HOST` | `host` | `qdrant.example.com` |
| `INDEXTER_STORE_PORT` | `port` | `6333` |
| `INDEXTER_STORE_GRPC_PORT` | `grpc_port` | `6334` |
| `INDEXTER_STORE_PREFER_GRPC` | `prefer_grpc` | `true` |
| `INDEXTER_STORE_API_KEY` | `api_key` | `your-api-key` |
| `INDEXTER_STORE_EMBEDDING_MODEL` | `embedding_model` | `BAAI/bge-small-en` |
| `INDEXTER_STORE_SPARSE_EMBEDDING_MODEL` | `sparse_embedding_model` | `Qdrant/bm25` |
| `INDEXTER_STORE_LOCAL_INFERENCE_BATCH_SIZE` | `local_inference_batch_size` | `16` |

### MCP (`INDEXTER_MCP_*`)

| Variable | Field | Example |
|----------|-------|---------|
| `INDEXTER_MCP_TRANSPORT` | `transport` | `http` |
| `INDEXTER_MCP_HOST` | `host` | `0.0.0.0` |
| `INDEXTER_MCP_PORT` | `port` | `9000` |

### Watch (`INDEXTER_WATCH_*`)

| Variable | Field | Example |
|----------|-------|---------|
| `INDEXTER_WATCH_ENABLED` | `enabled` | `true` |
| `INDEXTER_WATCH_DEBOUNCE_MS` | `debounce_ms` | `500` |
| `INDEXTER_WATCH_POLL_DELAY_MS` | `poll_delay_ms` | `3000` |

### XDG Directory Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `XDG_CONFIG_HOME` | `~/.config` | Base for config dir (`indexter.toml`, `repos.json`) |
| `XDG_DATA_HOME` | `~/.local/share` | Base for data dir (Qdrant storage) |
| `XDG_CACHE_HOME` | `~/.cache` | Base for cache dir |

### Common Recipes

```bash
# Use in-memory store for testing
INDEXTER_STORE_MODE=memory indexter serve

# Connect to a remote Qdrant instance
INDEXTER_STORE_HOST=qdrant.example.com INDEXTER_STORE_API_KEY=secret indexter serve

# Start MCP server over HTTP on a custom port
INDEXTER_MCP_TRANSPORT=http INDEXTER_MCP_PORT=9000 indexter serve

# Enable file watching
INDEXTER_WATCH_ENABLED=true indexter serve
```

## Directory Structure

Indexter follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/):

```
~/.config/indexter/          # XDG_CONFIG_HOME/indexter
├── indexter.toml            # Global configuration
└── repos.json               # Repository registry

~/.local/share/indexter/     # XDG_DATA_HOME/indexter
└── qdrant/                  # Qdrant vector store data

~/.cache/indexter/           # XDG_CACHE_HOME/indexter
```

## Default Ignore Patterns

Indexter ships with a comprehensive set of default ignore patterns. Per-repo patterns are **added** to these, not replacing them.

<details>
<summary>View all default patterns</summary>

| Category | Patterns |
|----------|----------|
| **Version Control** | `.git/`, `.git` |
| **System Files** | `.DS_Store`, `Thumbs.db` |
| **Python** | `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.env/`, `env/`, `*.egg-info/`, `.tox/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` |
| **Node.js** | `node_modules/`, `bower_components/`, `.next/`, `.nuxt/`, `.output/` |
| **Rust** | `target/` |
| **Build** | `dist/`, `build/`, `out/`, `bin/`, `obj/` |
| **Cache** | `.cache/`, `.temp/`, `.tmp/`, `tmp/`, `temp/` |
| **IDE/Editor** | `.idea/`, `.vscode/`, `.vs/` |
| **Dependencies** | `vendor/` |
| **Test Coverage** | `.coverage`, `coverage/`, `htmlcov/`, `.nyc_output/` |
| **Lock Files** | `*.lock`, `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `Cargo.lock`, `poetry.lock`, `uv.lock` |
| **Data Files** | `*.csv`, `*.sqlite`, `*.db`, `*.log`, `*.tsv`, `*.parquet`, `*.arrow`, `*.h5`, `*.hdf5` |

</details>

## Programmatic Access

```python
from indexter.config import settings, RepoSettings
from pathlib import Path

# Global settings singleton
print(settings.top_k)             # 10
print(settings.store.mode)        # "server"
print(settings.config_dir)        # ~/.config/indexter

# Per-repo settings (auto-loads from repo's indexter.toml or pyproject.toml)
repo = RepoSettings(path=Path("/path/to/my-repo"))
print(repo.collection_name)       # "indexter_my-repo"
print(repo.ignore_patterns)       # union of global + repo patterns

# All registered repos
repos = RepoSettings.load()
```
