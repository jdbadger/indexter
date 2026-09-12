## 1. Project scaffolding

- [x] 1.1 Rewrite `pyproject.toml` project metadata: name `indexter`, description, MIT license, author, keywords, classifiers, URLs, `requires-python = ">=3.11,<3.14"`
- [x] 1.2 Add runtime dependencies: `tree-sitter`, `tree-sitter-language-pack`, `sqlite-vec`, `sentence-transformers`, `fastmcp`, `typer`, `pydantic`, `pathspec`; add the single `onnx` extra pinning `fastembed`; no `core`/`cli`/`mcp`/`full` extras
- [x] 1.3 Add `dev` and `test` dependency groups mirroring the old repo (`ruff`, `ty`, `rust-just`, `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-testmon`, `inline-snapshot`, `dirty-equals`); no `pytest-asyncio`
- [x] 1.4 Configure `uv_build` backend with `module-root = "src"` and test exclusion; confirm `schema.sql` is included in the wheel
- [x] 1.5 Add `[project.scripts] indexter = "indexter.cli:app"`
- [x] 1.6 Port tool config: `[tool.ruff]` (line-length 120, select `E,F,I,UP,S,B`, per-file-ignores for tests), `[tool.ty]`, `[tool.pytest.ini_options]` with `--pyargs`, `--cov=indexter --cov-fail-under=95`
- [x] 1.7 Create the package skeleton `src/indexter/{__init__.py,config.py,paths.py,cli.py}` and `src/indexter/db/{__init__.py,connection.py,queries.py,schema.sql}`, plus empty `tests/` packages beside each module; delete `src/indexter/main.py`
- [x] 1.8 Add `.gitignore` entries for `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.db`, coverage artifacts
- [x] 1.9 Verify `uv sync --group test` resolves and `uv run indexter --help` executes the empty app

## 2. Paths (`paths.py`)

- [x] 2.1 Implement `data_dir()` and `config_dir()` honouring `XDG_DATA_HOME`/`XDG_CONFIG_HOME`, treating an empty value as unset, defaulting to `~/.local/share/indexter` and `~/.config/indexter`, with no directory creation as a side effect
- [x] 2.2 Implement `ensure_dir()` used by writers, so directories are created only when something is written
- [x] 2.3 Implement `canonical_repo_path()` resolving relative paths, `.`/`..`, trailing slashes, and symlinks
- [x] 2.4 Implement `slugify()` reducing a directory name to lowercase alphanumerics and hyphens, collapsing runs and trimming edges, with a fallback for a name that slugs to empty
- [x] 2.5 Implement `db_path(repo)` returning `data_dir() / f"{slug}-{sha256(canonical)[:12]}.db"`
- [x] 2.6 Tests: determinism across calls, distinct hashes for same-named repos in different parents, canonicalization equivalence for all four input forms, slug character set, XDG override and default, empty-string XDG, no directory created by path lookup

## 3. Configuration (`config.py`)

- [x] 3.1 Define the `Settings` pydantic model — frozen, `extra="forbid"`, typed fields with defaults (embedding model, embedding dimension, batch size, ignore patterns, max file size, snippet/limit budgets as needed by later milestones)
- [x] 3.2 Implement TOML loading with stdlib `tomllib`, raising an error that names the file path and the parse error
- [x] 3.3 Implement global-layer loading from `config_dir() / "config.toml"`, treating a missing file as an empty layer
- [x] 3.4 Implement repo-layer loading: `indexter.toml` at the repo root wins in full; otherwise `[tool.indexter]` from `pyproject.toml`; warn naming the winner when both exist
- [x] 3.5 Implement `load_settings(repo=None, **overrides)` merging defaults → global → repo → explicit overrides key by key, then validating once through `Settings`
- [x] 3.6 Map pydantic validation errors to messages naming the offending key, its source file, and the expected type
- [x] 3.7 Tests: defaults with no files, repo overrides global, partial override preserves other keys, explicit args win, missing global file is fine, malformed TOML errors, both repo sources present warns and uses `indexter.toml`, `[tool.indexter]` fallback, neither source present, unknown key rejected with file name, wrong type rejected, frozen model rejects assignment

## 4. Schema (`db/schema.sql`)

- [x] 4.1 Write `files`, `nodes`, `refs`, `edges`, and `project_metadata` DDL per the plan's column lists, with no foreign key constraints
- [x] 4.2 Add the `UNIQUE` constraints: `nodes.id`, and `edges(source, target, kind, IFNULL(line,-1))`
- [x] 4.3 Add the indexes later milestones will need: `nodes(file_path)`, `nodes(name)`, `nodes(kind)`, `nodes(parent_id)`, `refs(from_node_id)`, `refs(status)`, `refs(head)`, `edges(source, kind)`, `edges(target, kind)`
- [x] 4.4 Add the `nodes_fts` FTS5 virtual table with `id UNINDEXED`, `name`, `name_words`, `qualified_name`, `docstring`, `signature`, `body`, as a standalone (non-`content=`) table
- [x] 4.5 Keep the `vectors` vec0 DDL out of `schema.sql` — it is created programmatically in task 5.5
- [x] 4.6 Define `SCHEMA_VERSION` as a module constant in `db/connection.py`, starting at 1

## 5. Connection layer (`db/connection.py`)

- [x] 5.1 Define the typed error hierarchy: `IndexterDBError` with `ExtensionLoadingUnsupported`, `SqliteVecNotInstalled`, `SqliteVecLoadFailed`, `SchemaVersionMismatch`, `RepoPathMismatch` — each carrying the fields the CLI needs to render an actionable message
- [x] 5.2 Implement `_apply_pragmas()`: `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, `temp_store=MEMORY`, `foreign_keys=ON`; check the `journal_mode` return value and warn if WAL was declined
- [x] 5.3 Implement `_load_sqlite_vec()`: guard on `hasattr(conn, "enable_load_extension")`, import `sqlite_vec`, load, disable extension loading again, verify with `SELECT vec_version()`; map each of the three failure modes to its error with `sys.executable`/package name/loader message
- [x] 5.4 Implement schema creation: execute `schema.sql` read via `importlib.resources`, inside one transaction, writing to a temp path and renaming into place so a failure leaves no partial file
- [x] 5.5 Implement `create_vectors_table(conn, dim)` building the vec0 DDL from the configured dimension, and `rebuild_vectors_table(conn, dim)` dropping and recreating it
- [x] 5.6 Implement `project_metadata` read/write helpers with `updated_at`, writing `repo_path`, `model`, `dim`, `schema_version` at creation
- [x] 5.7 Implement `read_metadata(db_path)` that opens read-only without creating the schema, loading the extension, or version checking, so `list` works against any file
- [x] 5.8 Implement `open_db(...)` context manager: connect with `isolation_level=None` and `row_factory=sqlite3.Row`, apply pragmas, load and verify sqlite-vec, create-or-validate schema, check `schema_version` and `repo_path`, rebuild the vectors table on a dimension change, close on exit including on exception
- [x] 5.9 Tests — creation: all tables present after first open, second open leaves contents untouched, failed creation leaves no file
- [x] 5.10 Tests — round-trip: representative row per table with `NULL`s and JSON preserved, `nodes.id` uniqueness, edge uniqueness including the `IFNULL(line,-1)` case and the differing-line case
- [x] 5.11 Tests — FTS5: explicit-rowid insert, term search returns the rowid, join back to `nodes`
- [x] 5.12 Tests — vectors: KNN ordering, `kind`/`language` filters applied inside the KNN, wrong-length vector rejected
- [x] 5.13 Tests — missing FKs: edge to unknown node accepted; orphan-detection query returns exactly the orphans
- [x] 5.14 Tests — pragmas: values read back, transaction rollback discards rows, context manager closes on exception, WAL-declined warning path
- [x] 5.15 Tests — extension failures: all three modes simulated (monkeypatched `enable_load_extension` absence, `ImportError`, `OperationalError`), asserting the message content
- [x] 5.16 Tests — metadata: written at creation, version mismatch raises and leaves the file untouched, metadata readable despite mismatch or unloadable extension, repo path mismatch raises with both paths
- [x] 5.17 Tests — dimension change: vectors table rebuilt empty at the new dimension while `nodes`/`refs`/`edges`/`files` row counts are unchanged

## 6. Queries (`db/queries.py`)

- [x] 6.1 Add the metadata queries `list` needs: node count, last indexed time, model/dim/version lookup
- [x] 6.2 Add the orphan-detection maintenance queries for `refs` and `edges` used by tasks 5.13 and by M3/M4
- [x] 6.3 Tests for each query against a populated fixture database

## 7. CLI (`cli.py`)

- [x] 7.1 Create the Typer app with the project description as help text, and register `list` and `remove`; make `indexter` with no arguments print help and exit zero
- [x] 7.2 Implement `list`: glob `*.db` in the data directory, read each database's metadata without a full open, render a table of repo path, missing marker, node count, model, dimension, schema version, size, and last-indexed time
- [x] 7.3 Handle `list` edge cases: empty or absent data directory prints a message and exits zero; a schema-version mismatch shows the stored version; an unreadable file is reported as corrupt without aborting the listing
- [x] 7.4 Implement `remove`: accept a repo path or a database filename, resolve a repo path through `db_path()`, confirm unless `--yes`, delete the `.db` plus `-wal` and `-shm`, never touch the repository
- [x] 7.5 Handle `remove` edge cases: declined prompt deletes nothing and exits zero; no matching database exits non-zero with a clear message; a deleted repository whose database exists still removes cleanly
- [x] 7.6 Render `IndexterDBError` subclasses as one-line actionable messages with a non-zero exit, not tracebacks
- [x] 7.7 Tests via Typer's `CliRunner` with a temp `XDG_DATA_HOME`: help output and exit codes, unknown command, all `list` scenarios, all `remove` scenarios, error rendering

## 8. Verification

- [x] 8.1 `uv run --group dev ruff check --fix src/indexter` clean
- [x] 8.2 `uv run --group dev ty check src/indexter` clean
- [x] 8.3 `uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing` green
- [x] 8.4 Manual check: create a database for `~/dev/indexter`, confirm it lands at the derived path, `indexter list` shows it, `indexter remove` deletes it and its sidecars
- [x] 8.5 Manual check: run the CLI under an interpreter without SQLite extension support and confirm the error names the interpreter and the remedy
- [x] 8.6 Confirm `uv run --python 3.12` and `--python 3.13` pass the test suite alongside 3.11
