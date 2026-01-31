# MCP Server

Indexter exposes its indexing and search capabilities as an [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server built on [FastMCP 3.0](https://gofastmcp.com). AI agents connect over stdio or HTTP, register git repositories, build vector indexes, and run hybrid semantic + keyword searches — all through five tools.

## Table of Contents

- [Quick Start](#quick-start)
- [Tools](#tools)
  - [list\_repos](#list_repos)
  - [init\_repo](#init_repo)
  - [index\_repo](#index_repo)
  - [search\_repo](#search_repo)
  - [remove\_repo](#remove_repo)
- [Client Configuration](#client-configuration)
  - [Claude Desktop & Claude Code](#claude-desktop--claude-code)
  - [VS Code](#vs-code)
  - [Cursor](#cursor)
- [Lifecycle](#lifecycle)
- [CLI Usage](#cli-usage)
  - [Pre-Warming a Repository](#pre-warming-a-repository)
  - [Searching](#searching)
  - [Maintenance](#maintenance)
  - [Scripting](#scripting)
- [Configuration](#configuration)

## Quick Start

```bash
# Install (if not already)
uv tool install indexter

# Register and index a repo, then search it
# (all via your MCP client — e.g. Claude Desktop, VS Code Copilot, Cursor)
```

The server starts automatically when an MCP client connects. No manual startup needed for stdio mode.

For HTTP mode, set `transport` in your global config:

```toml
# ~/.config/indexter/indexter.toml

[mcp]
transport = "http"
host = "0.0.0.0"
port = 9000
```

Then start the server:

```bash
indexter-mcp
```

Or override at runtime with environment variables:

```bash
INDEXTER_MCP_TRANSPORT=http INDEXTER_MCP_PORT=9000 indexter-mcp
```

## Tools

### `list_repos`

List all registered repositories with metadata and index staleness.

**Parameters**: None

**Returns**: List of objects:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Repository name (derived from directory) |
| `path` | `str` | Absolute path to the repository root |
| `is_stale` | `bool` | Whether the index is out of date |
| `metadata` | `object` | Aggregate metadata (see below) |

**Metadata fields**:

| Field | Type | Description |
|-------|------|-------------|
| `documents` | `int` | Number of indexed documents (files) |
| `document_paths` | `list[str]` | Relative paths of indexed files |
| `nodes` | `int` | Total number of indexed code nodes |
| `node_types` | `list[str]` | Code construct types (e.g. `function`, `class`) |
| `languages` | `list[str]` | Programming languages detected |
| `document_tree` | `str` | ASCII tree representation of the file structure |

---

### `init_repo`

Register a new git repository with Indexter.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | `str` | Yes | Absolute path to a git repository root |

The repository must contain a `.git` directory. The name is derived from the directory name.

**Returns**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Repository name |
| `path` | `str` | Absolute path |
| `metadata` | `object` | Initial metadata |

---

### `index_repo`

Index (or re-index) a registered repository.

Performs **incremental indexing** by default — only changed files are re-indexed, detected via content hashing. Pass `full=True` to delete the existing index and rebuild from scratch.

**Parameters**:

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `name` | `str` | Yes | — | Repository name (as shown by `list_repos`) |
| `full` | `bool` | No | `false` | Full re-index instead of incremental |

**Returns** (`IndexResult`):

| Field | Type | Description |
|-------|------|-------------|
| `repo` | `str` | Repository name |
| `repo_path` | `str` | Absolute path |
| `documents_indexed` | `list[str]` | Files that were indexed |
| `documents_deleted` | `list[str]` | Files that were removed |
| `nodes_added` | `int` | New code nodes added |
| `nodes_deleted` | `int` | Code nodes removed |
| `indexed_at` | `datetime` | Indexing completion timestamp |
| `duration` | `float` | Operation duration in seconds |
| `errors` | `list[str]` | Any error messages |
| `summary` | `str` | Human-readable summary |

---

### `search_repo`

Semantic search over a repository's indexed code.

Combines **dense** (semantic) and **sparse** (keyword/BM25) search with Reciprocal Rank Fusion (RRF) for robust results.

**Parameters**:

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `name` | `str` | Yes | — | Repository name |
| `query` | `str` | Yes | — | Natural-language or code search query |
| `language` | `str` | No | `null` | Filter by language (e.g. `"python"`) |
| `node_type` | `str` | No | `null` | Filter by construct (e.g. `"function"`, `"class"`) |
| `node_name` | `str` | No | `null` | Filter by exact node name |
| `document_path` | `str` | No | `null` | Filter by file path (exact or directory prefix with trailing `/`) |
| `parent_scope` | `str` | No | `null` | Filter by enclosing scope (e.g. class name) |
| `has_documentation` | `bool` | No | `null` | Filter by presence of docstrings/comments |
| `limit` | `int` | No | `top_k` setting | Max results to return |

**Returns** (`SearchResults`):

| Field | Type | Description |
|-------|------|-------------|
| `repo` | `str` | Repository name |
| `repo_path` | `str` | Absolute path |
| `results` | `list[SearchResult]` | Matched nodes |
| `query` | `str` | Original query |
| `filters` | `dict` | Applied search filters |
| `count` | `int` | Number of results |

Each `SearchResult`:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Source code content |
| `score` | `float` | Relevance score (0.0–1.0) |
| `metadata` | `dict` | Node metadata (language, node\_type, node\_name, document\_path, start\_line, end\_line, parent\_scope, documentation, signature, etc.) |

---

### `remove_repo`

Remove a registered repository and delete its indexed data (collection, cache, and config entry). This is permanent.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | `str` | Yes | Repository name to remove |

**Returns**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Removed repository name |
| `removed` | `bool` | Whether the removal succeeded |

## Client Configuration

The MCP server entry point is `indexter-mcp`. Configure your client to launch it.

### Claude Desktop & Claude Code

Add to `claude_desktop_config.json`:

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

Add to `.vscode/settings.json` or user settings:

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

Add to Cursor MCP settings:

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

## Lifecycle

The server manages a complete lifecycle via FastMCP's `@lifespan` decorator:

1. **Startup**: Starts the Qdrant Docker container, waits for health check, creates a `QdrantClient`
2. **File watcher** (optional): If `watch.enabled = true`, spawns a background task that monitors registered repos for file changes and triggers incremental re-indexing
3. **Serve**: The `QdrantClient` is shared across all tool calls via FastMCP's lifespan context
4. **Shutdown**: Stops the file watcher (if running), closes the client, and stops the Qdrant container

> **Note**: The server requires `store.mode = "server"` (the default). Memory mode is only supported in tests.

## CLI Usage

[FastMCP's CLI](https://gofastmcp.com/cli/client) can connect to the Indexter server and call tools directly from the terminal — useful for debugging, scripting, and pre-warming indexes before your AI agent needs them.

The CLI connects via stdio by default. Point it at the server's module path:

```bash
# List available tools
fastmcp list indexter.mcp.server:server

# List with full input schemas
fastmcp list indexter.mcp.server:server --input-schema
```

> **Tip**: If you're running the server over HTTP, target the URL instead:
> ```bash
> fastmcp list http://localhost:8765/mcp
> fastmcp call http://localhost:8765/mcp list_repos
> ```

### Pre-Warming a Repository

Register a repo, build its index, and verify — all from the shell. This is useful to run before connecting an AI agent so searches are instant:

```bash
# 1. Register the repository
fastmcp call indexter.mcp.server:server init_repo path=/home/joe/dev/my-project

# 2. Build the full index
fastmcp call indexter.mcp.server:server index_repo name=my-project

# 3. Verify it's registered and not stale
fastmcp call indexter.mcp.server:server list_repos
```

To force a complete re-index (discard existing index and rebuild from scratch):

```bash
fastmcp call indexter.mcp.server:server index_repo name=my-project full=true
```

### Searching

```bash
# Basic semantic search
fastmcp call indexter.mcp.server:server search_repo name=my-project query="authentication middleware"

# Filter by language and node type
fastmcp call indexter.mcp.server:server search_repo \
  name=my-project \
  query="error handling" \
  language=python \
  node_type=function

# Filter by file path (directory prefix)
fastmcp call indexter.mcp.server:server search_repo \
  name=my-project \
  query="database connection" \
  document_path="src/db/"

# Find documented classes
fastmcp call indexter.mcp.server:server search_repo \
  name=my-project \
  query="data model" \
  node_type=class \
  has_documentation=true

# Get JSON output for piping to other tools
fastmcp call indexter.mcp.server:server search_repo \
  name=my-project \
  query="config" \
  limit=3 \
  --json
```

### Maintenance

```bash
# Check which repos need re-indexing
fastmcp call indexter.mcp.server:server list_repos

# Incrementally update a stale repo (only changed files)
fastmcp call indexter.mcp.server:server index_repo name=my-project

# Remove a repo and its indexed data
fastmcp call indexter.mcp.server:server remove_repo name=my-project
```

### Scripting

Combine `--json` output with `jq` for automation:

```bash
# Get names of all stale repos
fastmcp call indexter.mcp.server:server list_repos --json \
  | jq -r '.[] | select(.is_stale) | .name'

# Re-index all stale repos
for repo in $(fastmcp call indexter.mcp.server:server list_repos --json \
  | jq -r '.[] | select(.is_stale) | .name'); do
  echo "Re-indexing $repo..."
  fastmcp call indexter.mcp.server:server index_repo name="$repo"
done

# Pre-warm: init + index in one shot
fastmcp call indexter.mcp.server:server init_repo path=~/my-project \
  && fastmcp call indexter.mcp.server:server index_repo name=my-project
```

## Configuration

MCP server settings are in the `[mcp]` section of your global config. See the [Configuration README](../config/README.md) for the full reference.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transport` | `"stdio"` \| `"http"` | `"stdio"` | Transport mode |
| `host` | `str` | `"localhost"` | HTTP server hostname |
| `port` | `int` | `8765` | HTTP server port |

Environment variable overrides:

```bash
INDEXTER_MCP_TRANSPORT=http
INDEXTER_MCP_HOST=0.0.0.0
INDEXTER_MCP_PORT=9000
```
