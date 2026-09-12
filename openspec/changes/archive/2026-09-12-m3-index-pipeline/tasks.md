## 1. Configuration, dependencies, and scaffolding

- [x] 1.1 Create the `src/indexter/index/` package with `__init__.py` and co-located `tests/`
- [x] 1.2 Add `Settings.embedding_backend` as a `Literal["sentence-transformers", "fastembed"]` defaulting to `"sentence-transformers"`, and `Settings.embed_max_tokens` defaulting to 256
- [x] 1.3 Change the chunk fallback defaults to `chunk_size = 1000` and `chunk_overlap = 100`; update the chunk tests and regenerate the fallback snapshot
- [x] 1.4 Add `tokenizers` and `huggingface-hub` as direct dependencies in `pyproject.toml` and re-lock
- [x] 1.5 Tests: new keys have their defaults, an unknown backend name is rejected with a configuration error naming `embedding_backend`

## 2. Database layer changes (`db/connection.py`)

- [x] 2.1 Rebuild `vectors` in `open_db` when the stored model name *or* dimension differs from settings, updating both stored values
- [x] 2.2 Extract database-file deletion (the `.db`, `-wal`, `-shm` set) into a shared helper and use it from `remove`
- [x] 2.3 Tests: model change at the same dimension empties `vectors` and leaves `nodes`/`refs`/`edges`/`files` counts unchanged; unchanged model and dimension leave vectors in place; the deletion helper removes all three files and tolerates missing sidecars

## 3. Embedding (`index/embed.py`)

- [x] 3.1 Define the `Embedder` protocol: `model_name`, `tokenizer()`, `embed(texts) -> list[bytes]`
- [x] 3.2 Implement tokenizer loading from the model repository's `tokenizer.json` via `tokenizers` + `huggingface_hub` (local cache first, then download), with truncation and padding disabled, cached per instance
- [x] 3.3 Implement `SentenceTransformerEmbedder`: import sentence-transformers inside the first `embed()`, set `max_seq_length` to `embed_max_tokens`, batch by `embed_batch_size`, L2-normalize, serialize float32 for `vec0`; empty input returns `[]` without loading
- [x] 3.4 Implement `FastEmbedEmbedder` with the same contract, raising an error naming the `onnx` extra when `fastembed` isn't importable; confirm the default model name resolves under fastembed
- [x] 3.5 Verify the model's output dimension against `embedding_dim` on first load, raising a typed error naming both values and `embedding_dim`
- [x] 3.6 Wrap model/tokenizer download failures in a typed error naming the model and stating that one-time network access is required
- [x] 3.7 Implement `make_embedder(settings)` selecting the backend
- [x] 3.8 Implement `FakeEmbedder` for tests: whitespace tokenizer exposing the same encode/offsets surface, hash-derived unit vectors of the configured dimension, and counters for tokenizer and model loads
- [x] 3.9 Tests with stub `sentence_transformers`/`fastembed` modules: construction loads nothing, tokenizer loads without the model, model loads once across batches, output order/count/unit norm, dimension mismatch message, missing-fastembed message, download-failure message, backend selection
- [x] 3.10 Tests against the real cached tokenizer (skipped when not cached locally): the shipped `tokenizer.json`'s 128-token truncation does not cap counts
- [x] 3.11 One test against the real cached model (skipped when not cached locally): 384-dimensional unit vectors, semantically closer texts rank closer

## 4. Composition (`index/compose.py`)

- [x] 4.1 Implement identifier splitting (snake, kebab, camel, Pascal, acronym runs, digits) and path-word splitting
- [x] 4.2 Implement `qualified_name` and `name_words` for symbol nodes and for the file node
- [x] 4.3 Implement docstring splitting into prose and structured blocks for Google/NumPy (`Args`, `Returns`, `Raises`, …), JSDoc tags, and rustdoc `# Arguments`/`# Errors`/`# Examples` sections
- [x] 4.4 Implement the body section: slice by byte range, remove the already-emitted header and docstring, collapse blank-line runs, keep comments and string literals; fall back to the declaration header line when a node has no signature
- [x] 4.5 Implement child-residue extraction (a node's byte range minus its children's ranges) and use it for the FTS body of every node
- [x] 4.6 Implement the per-kind variants: container kinds list members by kind; file lists top-level symbols then residue; section uses heading/scope path and prose; data uses key path and a source slice; chunk uses path, line range, and raw text
- [x] 4.7 Implement token-budgeted truncation on the joined text: one encode without special tokens, cut at the end offset of the last in-budget token, budget `embed_max_tokens − 2`
- [x] 4.8 Implement `compose_file(relpath, content, parse_result, tokenizer, budget)` returning, per node ID, `embed_text`, `embed_hash`, `qualified_name`, `name_words`, and FTS `body`; define `INDEX_FORMAT_VERSION`
- [x] 4.9 Tests: identifier splitting table; section order for a documented method; body excludes docstring; literals kept; blank lines collapsed; JSDoc and rustdoc structured blocks placed last; each per-kind variant; FTS residue for method/class/file/script cases; reproducibility; line shift leaves text unchanged
- [x] 4.10 Tests — truncation with `FakeEmbedder`'s tokenizer: within-budget untouched, over-budget drops the structured block first and cuts the body, result re-tokenizes within budget
- [x] 4.11 Snapshot the composed text for every M2 fixture file with inline-snapshot, so a composer format change fails loudly and prompts an `INDEX_FORMAT_VERSION` bump

## 5. Sync — per-file writes (`index/sync.py`)

- [x] 5.1 Implement `write_file(conn, walked, content_hash, parse_result, composed)` in one `BEGIN IMMEDIATE` transaction: upsert the `files` row, upsert nodes on `id` preserving rowid, delete the file's nodes whose IDs vanished (with their FTS rows, vectors, and originating refs), replace the file's refs with status `unresolved`, replace FTS rows keyed by rowid, delete vectors whose `embed_hash` changed
- [x] 5.2 Implement `remove_file(conn, path)` in one transaction, deleting its `files` row, nodes, originating refs, FTS rows, and vectors
- [x] 5.3 Implement `record_unreadable(conn, walked, error)` storing an empty hash, zero nodes, and the error
- [x] 5.4 Implement `touch_file(conn, walked)` updating only size and mtime
- [x] 5.5 Tests: new file writes every column; re-sync of an unchanged node keeps rowid and vector; changed `embed_hash` drops the vector; removed symbol takes its FTS row, vector, and refs with it; refs are replaced not accumulated; FTS rowids equal node rowids; a failure injected mid-write rolls the file back to its previous state; `remove_file` leaves nothing behind (checked with the M1 orphan queries)

## 6. Sync — orchestration and embedding backlog

- [x] 6.1 Define `SyncReport`: added/changed/removed/unchanged paths, nodes written and deleted, refs written, texts embedded, per-file errors, elapsed seconds
- [x] 6.2 Implement the fingerprint (format version, `chunk_size`, `chunk_overlap`, `embed_max_tokens`, `embedding_model`) and its read/write in `project_metadata`
- [x] 6.3 Implement `sync_repo(conn, repo_path, settings, embedder)`: walk, load stored `files` rows once, classify by size+mtime, read and hash candidates, touch hash-equal files, parse and compose changed/new files (tokenizer requested only when something is composed), write per file, remove vanished paths, record read failures
- [x] 6.4 Implement the embedding backlog: select nodes lacking a vector, embed `embed_text` in `embed_batch_size` batches, insert per batch with kind and language, skip nodes that gained a vector concurrently; do not touch the embedder when the backlog is empty
- [x] 6.5 Write the fingerprint only after both passes complete
- [x] 6.6 Implement `index_repository(repo, settings, embedder, *, full=False)` for the CLI: resolve the database path, create or open it, rebuild on `SchemaVersionMismatch` (and when `full`), propagate `RepoPathMismatch`, run `sync_repo`, and report whether the database was created, rebuilt, or existed
- [x] 6.7 Tests with `FakeEmbedder` over a `tmp_path` repo: first index; immediate re-sync reports all unchanged, zero embedded, and zero tokenizer/model loads; touch reads one file and parses none; edit re-parses one file and embeds only changed texts; line insertion embeds nothing; delete, rename, and newly-ignored file removals; undecodable file recorded once and not re-read
- [x] 6.8 Tests — healing: backlog resumes after a simulated interruption without re-parsing; a model change re-embeds everything without re-parsing; a `chunk_size` change re-parses every file; a format-version bump with identical text keeps vectors; parse errors land in the report and `files.errors`
- [x] 6.9 Tests — `index_repository`: create, existing, `full` rebuild, schema-mismatch rebuild, repo-path mismatch propagated with the database untouched

## 7. CLI (`cli.py`)

- [x] 7.1 Add `indexter init [PATH]`: validate the path is a directory, load settings for the repository, run `index_repository`, print the summary (noting "already initialized" when it existed, "rebuilt" when it was rebuilt) and per-file errors
- [x] 7.2 Add `indexter reindex [PATH] [--full]`: exit non-zero pointing at `init` when no database exists; otherwise run `index_repository`, printing the summary
- [x] 7.3 Report `IndexterDBError`, `ConfigError`, and embedder errors as one-line messages with a non-zero exit
- [x] 7.4 Tests via `CliRunner` with `FakeEmbedder` injected: init creates and summarizes; init on existing syncs and says so; init on a non-directory exits non-zero and creates nothing; reindex without a database suggests init; reindex no-op summary; `--full` rebuilds; schema-mismatch rebuild message; repo-path mismatch exits non-zero; parse errors listed with exit zero; `--help` lists the new commands

## 8. Verification

- [x] 8.1 `uv run --group dev ruff check --fix src/indexter` clean
- [x] 8.2 `uv run --group dev ty check src/indexter` clean
- [x] 8.3 `uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing` green
- [x] 8.4 Real-repo check: `indexter init ~/dev/indexter` completes with the real model; record file, node, ref, and embedding counts and wall time (model load vs. the rest)
- [x] 8.5 Real-repo check: `indexter reindex ~/dev/indexter` immediately after reports all files unchanged, zero embedded, and does not import torch
- [x] 8.6 Real-repo check: touch one file → reindex reads it and parses nothing; edit one function → reindex re-parses only that file and embeds only the changed texts
- [x] 8.7 Real-repo check: in-process `sync_repo` with a warm embedder after a one-function edit completes well under a second; record the time
- [x] 8.8 Real-repo check: the M1 orphan queries return nothing after the above; `indexter list` shows the node count and last-indexed time
- [x] 8.9 Inspect a sample of `embed_text` values (a method, a class, a file node, a markdown section, a data block) and confirm they read as intended and fit the budget
- [x] 8.10 Confirm the suite passes on Python 3.11, 3.12, and 3.13
