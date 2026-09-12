## Why

M1 built an empty database and M2 produces `ParsedNode`s and `ParsedRef`s in memory; nothing yet connects them. M3 is the pipeline between: compose each node into embeddable text, embed it, and write nodes, refs, FTS rows and vectors — then keep that database current cheaply enough that settled decision 5 (sync on every search, no bypass) is affordable. Until this lands there is nothing for M4 to resolve or M5 to search.

## What Changes

- Add `index/compose.py`: deterministic embed-text composition, no LLM. Sections ordered most-meaningful-first — label (kind, qualified name, identifiers split into words, path words), signature, docstring prose, body prefix with string literals kept, and structured parameter/return/raise documentation last — truncated to the model's token budget with the model's own tokenizer, at a token boundary. Per-kind variants for classes (member names instead of bodies), files (top-level symbols), markdown sections (heading path + prose), data blocks (key path + slice), and chunks.
- Add `index/embed.py`: an `Embedder` interface with a **sentence-transformers** backend (default, torch imported lazily) and a **fastembed** backend (the `onnx` extra). The tokenizer loads separately from the model, so work that only needs to count tokens never imports torch. The model's actual output dimension is checked against configuration and a mismatch fails loudly.
- Add `index/sync.py`: incremental synchronization of one repository into its database.
  - Change detection in three steps: size+mtime against `files`, then content hash, then re-parse — only files whose hash changed are parsed.
  - Per-file atomic writes of `files`, `nodes`, `refs` and `nodes_fts`; nodes are upserted on their stable ID so their rowids — and therefore their vectors — survive a re-sync when their embed text is unchanged.
  - Files that vanish from the walk (deleted, newly ignored, now oversized) are removed with everything derived from them.
  - Embedding is a separate backlog pass over nodes that have no vector, so an interrupted run, a model change, or a dimension rebuild all heal on the next sync without re-parsing.
  - An index fingerprint (composer/parser format version plus the settings that shape parsing and composition) forces a re-parse of every file when it changes — without re-embedding texts that come out identical.
  - A sync report (added/changed/removed/unchanged files, nodes written, texts embedded, per-file errors) that M4 will consume to scope resolution.
- Add CLI commands `indexter init [PATH]` (create and fully index) and `indexter reindex [PATH] [--full]` (incremental sync; `--full` rebuilds from scratch). Both rebuild automatically when the stored schema version is out of date, as M1's design deferred to them.
- Change the connection layer so a changed **embedding model** — not only a changed dimension — rebuilds the `vectors` table. Two models with the same dimension produce incomparable vectors; today that case silently mixes them.
- Resolve M2's open question on chunk sizing: the fallback's defaults move from 250/25 to 1000/100 bytes, roughly one 256-token window at the ~3.5 bytes/token measured on this repo's code and prose.

## Capabilities

### New Capabilities
- `embed-text-composition`: What text represents each node for embedding — section order, identifier splitting, docstring handling, per-kind variants, and token-budgeted truncation.
- `embedding`: Turning composed text into vectors — backend selection, lazy model and tokenizer loading, normalization, batching, and dimension verification.
- `index-sync`: Keeping a repository's database current — change detection, per-file atomic writes, removal of vanished files, vector reuse and the embedding backlog, the index fingerprint, and the sync report.

### Modified Capabilities
- `database-schema`: the vectors-rebuild requirement widens from "dimension change" to "model or dimension change".
- `repo-management-cli`: adds the `init` and `reindex` commands, including automatic rebuild of out-of-date databases.

## Impact

- **New code**: `src/indexter/index/` (`__init__.py`, `compose.py`, `embed.py`, `sync.py`) with co-located tests; `init` and `reindex` in `cli.py`.
- **Changed code**: `db/connection.py` (model-change rebuild); `config.py` gains `embedding_backend` and `embed_max_tokens`, and the chunk defaults change.
- **Schema**: no DDL change and no schema-version bump — M1's tables already have every column M3 writes. `project_metadata` gains an `index_fingerprint` key, written by sync.
- **Dependencies**: `tokenizers` and `huggingface-hub` become direct dependencies (both already installed transitively by sentence-transformers and fastembed) because the composer loads the tokenizer without the model.
- **Not touched**: `edges` and `nodes.degree` — derived from refs in M4. M3 writes refs with status `unresolved` and leaves resolution alone.
- **Downstream**: M4 resolves the refs written here, scoped by the sync report; M5 searches the vectors and FTS rows written here, and relies on sync being fast enough to run before every query.
- **First run needs the model**: `init` downloads the embedding model into the Hugging Face cache if it is not already there. Later runs work offline.
