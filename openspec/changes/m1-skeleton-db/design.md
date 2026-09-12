## Context

`indexter` v0.1.2 stored vectors in a Docker-hosted Qdrant instance and tracked repos in a `repos.json` registry that could disagree with the collections it named. The rewrite collapses all storage — embeddings, FTS5 keyword index, and code graph — into one SQLite file per repo, held centrally and addressed by a path derived from the repo itself.

M1 builds only the substrate: dependency set, configuration, path derivation, schema, connection lifecycle, and the two CLI commands that need nothing more than the file layout (`list`, `remove`). No walking, parsing, embedding, or search. Every later milestone writes through the connection layer defined here, so the schema shape and the failure modes need to be right now rather than patched over in M3.

Constraints carried in from the plan: synchronous throughout; `sqlite-vec` with no numpy fallback; one database per repo at `~/.local/share/indexter/<name>-<hash12>.db`; no registry file; existing repo conventions (uv, ruff, ty, pytest ≥95% coverage, tests co-located as `<module>/tests/`).

## Goals / Non-Goals

**Goals:**
- A repo path deterministically names exactly one database file, with no registry to drift.
- Opening a database yields a connection with pragmas applied, `sqlite-vec` loaded and verified, and the schema present — or a typed error that says exactly what to do about it.
- The full v1 schema exists and round-trips: every table, index, and virtual table that M2–M5 will write to, so no later milestone has to change table shapes.
- Configuration resolves from packaged defaults, a global user file, and a per-repo file, with unknown keys rejected rather than silently ignored.
- `indexter list` and `indexter remove` operate over the data directory alone, reading each database's own `project_metadata` — they never consult a registry.

**Non-Goals:**
- Walking, parsing, embedding, resolution, search, MCP (M2–M6).
- The `init`, `reindex`, `mcp`, and `skill` commands beyond whatever registration the Typer app needs.
- In-place schema migrations (see Decision 4).
- Concurrency beyond "one writer at a time via WAL and a busy timeout".
- Cross-repo or multi-database queries.

## Decisions

### 1. Database path derived from the canonical repo path, not recorded

`db_path(repo) = data_dir() / f"{slug(canonical.name)}-{sha256(str(canonical))[:12]}.db"`, where `canonical = Path(repo).resolve()` and `data_dir()` honours `XDG_DATA_HOME`, defaulting to `~/.local/share/indexter`.

*Why:* the v0.1.2 failure mode was a registry that disagreed with reality. A pure function from repo path to file path cannot disagree with anything; the mapping is recomputable from either direction (the reverse lookup lives in each database's own `project_metadata.repo_path`).

*Alternatives:* (a) database inside the repo as `.indexter/index.db` — pollutes the working tree, needs a `.gitignore` entry per repo, and is lost on a fresh clone; (b) a registry file mapping repo → database — the thing being removed; (c) full path hash with no readable prefix — correct but makes `ls ~/.local/share/indexter` useless to a human.

*Consequences:* 12 hex characters is 48 bits — collision risk across the hundreds of repos this will ever hold is negligible, and a collision is caught anyway because `project_metadata.repo_path` won't match. Two spellings of the same directory on a case-insensitive filesystem produce two databases; we deliberately do not case-fold, since that would be wrong on Linux. Moving a repo orphans its database — `list` reports it as missing and `remove` cleans it up.

### 2. `sqlite-vec` is mandatory and its absence is a startup failure with three distinct messages

After `sqlite3.connect`, the connection layer calls `enable_load_extension(True)`, `sqlite_vec.load(conn)`, `enable_load_extension(False)`, then verifies with `SELECT vec_version()`. Three failure modes get three messages:

| Failure | Detection | Message |
|---|---|---|
| Interpreter built without extension support | `hasattr(conn, "enable_load_extension")` is `False`, or it raises | Name the interpreter (`sys.executable`) and tell the user to run under a uv-managed Python |
| `sqlite_vec` not installed | `ImportError` | Name the package and the install command |
| Extension binary won't load | `sqlite3.OperationalError` from `load()` | Surface the underlying loader error verbatim plus the platform/arch |

*Why:* this is verified, not hypothetical — on this machine `python3 -c "import sqlite3; hasattr(sqlite3.connect(':memory:'), 'enable_load_extension')"` returns `False` for the system interpreter and `True` under `uv run --python 3.11` (CPython 3.11.15, SQLite 3.50.4, FTS5 present). A user who hits this with a generic "no such module: vec0" gets no clue what to do.

*Alternative rejected:* a numpy brute-force fallback. It would silently turn a 3 ms KNN into a linear scan and give the project two search paths to test forever, for a dependency that is a pure-wheel install on every platform we target.

### 3. The vector table is created separately from `schema.sql`, parameterized by dimension

`schema.sql` holds everything whose DDL is fixed: `files`, `nodes`, `refs`, `edges`, `nodes_fts`, `project_metadata`, and indexes. The `vectors` virtual table embeds its dimension in the DDL (`emb float[384]`), so it is created by a function that takes the dimension from config — and, when `project_metadata.embedding_dim` no longer matches, dropped and recreated on its own.

*Why:* the plan requires that a model change rebuild vectors without re-parsing. Keeping that one table's DDL under program control makes that a five-line operation instead of a schema migration.

*Alternative rejected:* string-templating the whole `schema.sql`. It turns a readable, lintable SQL file into a format string for the sake of one integer.

### 4. The database is a derived cache: schema evolution is rebuild, not migrate

`project_metadata.schema_version` is written at creation. On open, a mismatch raises `SchemaVersionMismatch` carrying the database path, the found version, and the expected one. The connection layer never rewrites or deletes on its own; the commands that own rebuilding (`init`/`reindex`, M3) catch it and rebuild from source. `list` and `remove` read metadata without a full open, so an out-of-date database is still listable and removable.

*Why:* every byte in the database is recomputable from the repo in seconds (measured: ~1–2 s full index of a 2,500-node repo). Hand-written migration scripts for a rebuildable cache are pure cost, and a wrong one corrupts a graph silently.

*Alternative rejected:* numbered forward-only migration scripts. Justified for user data; not for an index that is cheaper to rebuild than to migrate carefully.

*Consequence:* `schema_version` bumps on any DDL change, including additive ones. That is intentional — it keeps the rule to one branch.

### 5. No foreign key constraints in the schema

`refs.from_node_id`, `edges.source`, and `edges.target` hold stable text node IDs and are deliberately allowed to dangle. Extraction (M2) writes refs before the targets exist; resolution (M4) is a second pass; incremental sync (M3) rewrites one file's nodes while other files' edges still point at them mid-transaction.

*Why:* the two-pass design and per-file incremental sync are fundamentally at odds with referential integrity enforced per-statement. Integrity is a property of the sync transaction, checked by query, not by constraint.

*Consequence:* orphan detection becomes a maintenance query rather than a guarantee. `PRAGMA foreign_keys=ON` is still set so that any FK added later behaves as expected.

### 6. Two ID spaces, wired explicitly

`nodes.id` (stable text) is the graph's currency; `nodes.rowid` (unstable integer) keys `vectors.node_rowid` and `nodes_fts.rowid`. `nodes_fts` is a standalone FTS5 table (not `content=`-backed), because bodies live only there; rows are inserted with an explicit `rowid` equal to the node's, and the `id` column is `UNINDEXED` so a hit joins straight back to `nodes`.

*Why:* re-indexing one file must not invalidate another file's edges (hence stable text IDs), while `vec0` and FTS5 both want integer rowids for fast joins. The mapping is maintained per-file on sync — the cost is one delete-and-reinsert per changed file, which is already happening.

### 7. Configuration: three layers, plain pydantic, `tomllib`, unknown keys rejected

Precedence, lowest to highest: packaged defaults → `~/.config/indexter/config.toml` (XDG-aware) → per-repo config → explicit call arguments. The per-repo layer is `indexter.toml` at the repo root if present, otherwise `[tool.indexter]` in the repo's `pyproject.toml`; the two are not merged, and finding both emits a warning naming the one that won. Models are plain `pydantic.BaseModel` with `extra="forbid"`; TOML is parsed with stdlib `tomllib`.

*Why:* `extra="forbid"` turns a typo'd key from a silently ignored setting into an error at the moment the file is read. Not merging the two repo-level sources keeps the mental model to one sentence. Dropping `pydantic-settings` and `tomli`/`tomlkit` (all present in v0.1.2) removes three dependencies that 3.11's `tomllib` and a hand-written 30-line layering function cover.

*Alternative rejected:* environment-variable overrides for every setting via `pydantic-settings`. Only the XDG variables are honoured; a per-setting env layer is surface area no one asked for.

### 8. Connection lifecycle: explicit transactions, one connection per caller

`sqlite3.connect(path, isolation_level=None)` with `row_factory = sqlite3.Row`, exposed as a context manager that yields the connection and closes it. `isolation_level=None` means Python issues no implicit `BEGIN`; the sync layer takes explicit `BEGIN IMMEDIATE` for its write batches. Pragmas: `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, `temp_store=MEMORY`, `foreign_keys=ON`. `journal_mode` is verified from its return value, since it silently declines on some filesystems.

*Why:* implicit transaction handling in `sqlite3` is the classic source of "why is my index half-written" — the index pipeline wants one transaction per file sync, chosen by the pipeline. Connections are cheap; a shared cache would need a lock discipline that buys nothing at this scale. M6 will open one connection per MCP worker thread.

### 9. `list` and `remove` read the data directory, and each database's own metadata

`list` globs `*.db`, opens each read-only (no schema creation, no `sqlite-vec` needed — `project_metadata` is an ordinary table), and reports repo path, whether that path still exists, node count, model, dimension, schema version, file size, and last index time. A file that is not a readable indexter database is reported as corrupt rather than skipped.

`remove` accepts a repo path (canonicalized through the same derivation) or a database filename, confirms unless `--yes`, and deletes the `.db` plus its `-wal` and `-shm` sidecars.

*Why:* keeping these two commands off the full-open path means a database from a future or past schema version, or one whose `sqlite-vec` won't load, is still inspectable and removable. That is exactly when you need them.

## Risks / Trade-offs

- **`sqlite-vec` unavailable on a user's interpreter** → verified failure with a message naming the interpreter and the fix; documented in the README as a hard requirement. Confirmed to work on uv-managed CPython 3.11–3.13, which is the supported install path.
- **Schema churn across M2–M5 forcing rebuilds** → the schema is written in full now from the plan's table definitions rather than grown milestone by milestone, so churn should be limited to indexes. Rebuild is cheap by design (Decision 4), so the blast radius of being wrong is a re-index, not a data-loss migration.
- **Dangling text IDs with no FK enforcement (Decision 5)** → integrity checked by query in the sync layer's tests; a small set of orphan-detection queries lands in `db/queries.py` in M1 so M3/M4 have them from the start.
- **WAL declined on a network filesystem** → return value of `PRAGMA journal_mode` is checked and a warning is emitted; the database still works, more slowly. Central storage under `~/.local/share` makes this unlikely.
- **Orphaned databases after a repo move or delete** → not automatically reaped; `list` marks the repo path missing and `remove` cleans up. Automatic reaping would be a footgun for a repo on an unmounted volume.
- **Case-insensitive filesystems creating duplicate databases for one repo** → accepted, and cheap to hit only by typing a differently-cased path. Case-folding would be incorrect on Linux.
- **`extra="forbid"` on config breaks forward compatibility** → an older indexter reading a newer config file errors. Accepted: for a single-user local tool, a loud unknown-key error is worth more than tolerant parsing.

## Migration Plan

There is nothing deployed to migrate — this is the first change in a new repository. For a user coming from `indexter` v0.1.2, the two are separate packages with separate storage; the old Qdrant containers and `repos.json` are untouched and can be removed by hand. The README will say so in M6/M7.

## Open Questions

- **Embedding model and dimension defaults.** The schema and `project_metadata` are written for 384 dimensions (MiniLM); the exact default model string is settled in M3 when embedding lands. M1 stores whatever config supplies and enforces only that it matches on reopen.
- **Whether `indexter remove` should also accept a bare repo name.** Deferred until `list`'s output shows how people actually refer to these.
- **`db/queries.py` scope.** M1 adds only the metadata and maintenance queries `list`/`remove` and the integrity checks need; whether the rest of the project's SQL centralizes here or lives beside its callers is decided in M3 when there is real traffic.
