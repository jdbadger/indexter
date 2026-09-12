## Why

The indexter rewrite replaces a Docker-hosted Qdrant deployment with a single SQLite file per repo holding embeddings, an FTS5 keyword index, and a code graph. Nothing else can be built until that file exists, is reachable at a deterministic path, and opens with the right pragmas and the sqlite-vec extension loaded. M1 is the foundation the parse, index, resolve, and search milestones all write into.

## What Changes

- Declare the real dependency set in `pyproject.toml` (`tree-sitter`, `tree-sitter-language-pack`, `sqlite-vec`, `sentence-transformers`, `fastmcp`, `typer`, `pydantic`, `pathspec`) with a single `onnx` extra for `fastembed`. **BREAKING** relative to the old repo: the `core`/`cli`/`mcp`/`full` extras split goes away.
- Add layered configuration: packaged defaults ← `~/.config/indexter/config.toml` ← per-repo `indexter.toml` or `[tool.indexter]` in the repo's `pyproject.toml`. No `repos.json` registry — **BREAKING** relative to v0.1.2.
- Add deterministic storage-path derivation: a canonical repo path maps to exactly one database at `~/.local/share/indexter/<name>-<hash12>.db`, honouring `XDG_DATA_HOME` / `XDG_CONFIG_HOME`.
- Add `db/schema.sql` with the full v1 schema — `files`, `nodes`, `refs`, `edges`, `nodes_fts` (FTS5), `vectors` (vec0, 384-dim), `project_metadata` — plus the indexes search will need.
- Add a connection layer that applies pragmas (WAL, foreign keys, busy timeout, synchronous NORMAL), loads sqlite-vec, creates the schema on first open, and checks `schema_version` on subsequent opens.
- Fail loudly and actionably when sqlite-vec cannot be loaded (missing extension support in the interpreter, or missing package) rather than silently degrading — there is no numpy fallback.
- Record repo path, embedding model, embedding dimension, and schema version in `project_metadata` at creation; detect model/dimension mismatch on open.
- Add the `indexter list` and `indexter remove` CLI commands over the central database directory, replacing `main.py`'s placeholder with a Typer app.

## Capabilities

### New Capabilities
- `configuration`: Layered config resolution (packaged defaults, global user config, per-repo config), typed settings model, and precedence/validation rules.
- `storage-paths`: Deterministic mapping from a repo path to its database file, plus the XDG-aware config and data directory locations.
- `database-schema`: The SQLite schema, connection lifecycle, pragmas, sqlite-vec extension loading, schema versioning/migration, and `project_metadata` bookkeeping.
- `repo-management-cli`: The `indexter list` and `indexter remove` commands and the shared CLI entry point they hang off.

### Modified Capabilities
<!-- None. No specs exist yet; this is the first change in the repo. -->

## Impact

- **New code**: `src/indexter/config.py`, `src/indexter/paths.py`, `src/indexter/db/{__init__.py,schema.sql,connection.py,queries.py}`, `src/indexter/cli.py`, with co-located tests under `tests/`.
- **Removed code**: `src/indexter/main.py` placeholder.
- **Packaging**: `pyproject.toml` gains dependencies, the `onnx` extra, a `[project.scripts]` `indexter` entry point, and package-data inclusion for `schema.sql`.
- **Runtime requirement**: the Python interpreter must support SQLite extension loading (`sqlite3.Connection.enable_load_extension`). uv-managed interpreters do; some system Pythons (notably macOS system Python) do not. This becomes a documented startup failure, not a silent one.
- **Downstream**: M2–M7 all write through this connection layer; the schema shape and node ID conventions defined here are assumed by the index, resolve, and search milestones.
- **Not in scope**: walking, parsing, embedding, resolution, search, and the MCP server. `init`, `reindex`, `mcp`, and `skill` commands are stubbed only insofar as the CLI app needs a shape; their behavior lands in later milestones.
