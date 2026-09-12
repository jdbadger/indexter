## Context

M1 left a database with every table the plan needs and a connection layer that validates it; M2 turns one file into `ParsedNode`s and `ParsedRef`s with stable IDs and parent links. Nothing yet writes those into the database, embeds anything, or notices when a file changes. M3 builds that pipeline and the two CLI commands that drive it.

The hard constraint is settled decision 5: **search syncs first, every time, with no bypass**. The pipeline is therefore not a batch job that runs occasionally — its no-change path runs before every query, and its one-file-changed path runs whenever an agent edits code between searches. The M3 success criteria encode this: index `~/dev/indexter`, a re-run is a no-op, one edited file syncs well under a second.

Measured on this machine before writing this (all with the M1 database layer and the cached `all-MiniLM-L6-v2`):

| Measurement | Result |
|---|---|
| `tokenizers` load of the model's `tokenizer.json`, no torch | 0.11 s |
| sentence-transformers model load, warm file cache | 1.94 s (26.9 s cold) |
| Embed 10 / 2,500 short texts (MPS) | 5 ms / 0.47 s |
| Bytes per token — Python / Markdown / TOML | 3.66 / 3.40 / 2.31 |
| Nodes lacking a vector, over 20,000 nodes (`rowid NOT IN (SELECT node_rowid FROM vectors)`) | 3.6 ms |
| `INSERT … ON CONFLICT(id) DO UPDATE` on `nodes` | preserves `rowid` |
| `INSERT OR REPLACE` into a `vec0` table | rejected: *"UNIQUE constraint failed on vectors primary key"* |
| Model's `tokenizer.json` as shipped | truncation at 128 tokens and padding **enabled** |

The last row matters: counting tokens with the tokenizer as loaded would silently cap every count at 128, so the composer would believe a 400-token text fits a 256-token budget.

Constraints from the plan and earlier milestones: synchronous throughout; `refs` are the durable source of truth and edges are derived from them in M4; node bodies are never stored in `nodes`, only in `nodes_fts`; vectors and FTS key off `nodes.rowid`; the database is a rebuildable cache (M1 decision 4); tests co-located, ≥95% coverage.

## Goals / Non-Goals

**Goals:**
- A full index of a repository the size of `~/dev/indexter` in a few seconds, dominated by model load.
- A sync with nothing changed reads no file contents, loads no tokenizer and no model, and writes nothing.
- A sync after one file changed re-parses that file only and re-embeds only the nodes whose composed text actually changed — well under a second with a warm embedder.
- Every interruption — killed process, model change, dimension change, format change — heals on the next sync without a manual rebuild.
- Composed text that is deterministic, token-budgeted with the model's own tokenizer, and inspectable (stored in `nodes.embed_text`).
- `indexter init` and `indexter reindex` as the setup-time entry points.

**Non-Goals:**
- Resolving refs, writing `edges`, computing `nodes.degree`, the builtin stoplist, `external_module` nodes — all M4.
- Query embedding, ranking, snippets — M5 (the `Embedder` built here is what M5 will call).
- Holding a warm embedder across requests — that is the MCP server's job in M6.
- A filesystem watcher, parallel parsing, progress bars.
- More than one vector per node (see Open Questions on long markdown sections).
- LLM-written summaries (out of scope for v1).

## Decisions

### 1. Sync is two passes: structural writes, then an embedding backlog

Pass one walks, detects changed files, parses and composes them, and writes `files`, `nodes`, `refs` and `nodes_fts`. It never calls the model. Pass two selects every node that has no row in `vectors`, embeds their `embed_text` in batches, and inserts the vectors.

*Why:* a node "needs embedding" is then a fact readable from the database, not a list carried in memory. That single rule covers the first index, a changed file, a killed process halfway through embedding, a model change that emptied `vectors`, and a dimension rebuild — all of them are "nodes without vectors", and all of them heal on the next sync with no special cases. The backlog query is 3.6 ms over 20,000 nodes.

*Consequence:* between the two passes a node is searchable by keyword but not by vector. Search syncs in-process before querying (decision 5), so it never observes that state except after a crash, and then only until the next sync.

*Alternative rejected:* parse, compose and embed per file, writing everything together. Simpler to read, but a crash mid-embed leaves nodes that are indistinguishable from embedded ones unless extra state is tracked, and a model change would need its own re-embed code path.

### 2. Nodes are upserted on their stable ID, so rowids — and vectors — survive re-syncs

Writing a changed file upserts each node with `INSERT … ON CONFLICT(id) DO UPDATE`, which keeps the existing `rowid`; nodes of that file whose IDs are no longer produced are deleted along with their FTS row and vector. For a surviving node, the vector is deleted only when the new `embed_hash` differs from the stored one.

*Why:* the plan keys vectors off `rowid` and promises re-embedding "only when the text actually changes". Upsert makes both true at once: an edit to one function in a 40-node file leaves 39 rowids and 39 vectors untouched, and pass two embeds one or two texts. Line shifts change `start_line`/`start_byte` but not the composed text, so they re-embed nothing.

*Alternative rejected:* delete every node in the file and re-insert, copying vectors across by `embed_hash`. It works, but it churns rowids on every edit, reads and rewrites vector blobs that didn't change, and needs a staging step because `vec0` rejects `INSERT OR REPLACE` (measured).

### 3. One transaction per file, not per sync

Each changed file's writes — its `files` row, node upserts and deletions, ref replacement, FTS replacement, vector invalidation — run inside one `BEGIN IMMEDIATE … COMMIT`. Removing a vanished file is likewise one transaction. Backlog inserts commit per batch.

*Why:* a first index of a large repository survives interruption with every completed file intact, and a reader under WAL always sees each file either wholly old or wholly new. `synchronous = NORMAL` in WAL mode makes commits cheap. `files.content_hash` is written in the same transaction as the nodes derived from it, so a file is never recorded as indexed at a hash whose nodes aren't there.

*Consequence:* a reader can see a sync half-applied across files. Nothing in M3 reads across files; M4's resolver runs after pass one and is scoped by the sync report (decision 10).

### 4. Change detection: size and mtime, then hash, then parse

For each walked file, a stored `files` row with the same size and mtime means unchanged — no read. Otherwise the file is read and hashed (`sha256(relpath:content)`, from M2); an unchanged hash updates only the stored size and mtime. Only a changed hash (or a new file) is parsed. Walked stat values are the ones stored, taken before the read, so a modification that lands after the read still differs next time.

A file that can't be decoded is recorded with an empty hash, zero nodes and an error, so it is not re-read on every sync until its stat changes. Stored files absent from the walk — deleted, newly ignored, now oversized or binary — are removed with every node, ref, FTS row and vector derived from them.

*Why:* this is settled decision 5 verbatim, and it is what makes the no-op path cost a walk plus one `SELECT path, size, mtime FROM files`.

*Alternative rejected:* hash every file on every sync. Always correct, but it reads the whole repository before every search.

### 5. An index fingerprint forces a re-parse when format or settings change

`project_metadata.index_fingerprint` stores a hash of an `INDEX_FORMAT_VERSION` constant (bumped whenever parser output or composer format changes), `chunk_size`, `chunk_overlap`, `embed_max_tokens`, and `embedding_model`. When it differs from the current value, every walked file is treated as changed. The fingerprint is written only after a sync completes.

*Why:* file-level change detection can't see that the *code* producing nodes changed, or that a setting reshaping chunks or truncation changed. Re-parsing is cheap; re-embedding is what costs, and decision 2 means texts that compose identically keep their vectors anyway. The model is included because its tokenizer decides where truncation falls.

*Consequence:* forgetting to bump `INDEX_FORMAT_VERSION` after changing the composer leaves stale `embed_text` until files change. A test pins the composer's output over the fixtures, so a format change fails that test and prompts the bump.

### 6. Composition works on a whole file, not one node at a time

`compose_file(relpath, content, parse_result, tokenizer, budget)` produces, per node, `embed_text`, `name_words`, `qualified_name`, and the FTS `body`. It needs the file because class and file nodes describe their children, and bodies are sliced from the source by byte range.

Section order, most meaningful first, joined with newlines:

1. **Label** — kind, qualified name, the name split into words, the path split into words. `method AuthHandler.login | auth handler login | src/auth/handlers.py (src auth handlers)`.
2. **Signature**, when the node has one; otherwise its declaration header line.
3. **Docstring prose** — everything except structured blocks.
4. **Body prefix** — the node's source with the header and docstring already emitted removed, runs of blank lines collapsed, comments and string literals kept.
5. **Structured documentation** — Google/NumPy-style `Args`/`Returns`/`Raises` sections, JSDoc `@param`/`@returns`/`@throws` tags, rustdoc `# Arguments`/`# Errors`/`# Examples` sections.

Per-kind variants replace section 4: **class/struct/trait/interface/enum** list member names by kind instead of bodies; **file** lists its top-level symbols, then its residue (imports, module-level statements); **section** (markdown/HTML/CSS) uses its heading or scope path as the label and its prose as the body; **data** uses its key path and a source slice; **chunk** uses the path, line range and raw text.

Identifier splitting handles snake_case, camelCase, PascalCase, acronym runs and digits: `get_user_by_email` → `get user by email`, `HTTPServer2` → `http server 2`. `qualified_name` is the scope path and name joined with `.`; for the file node it is the relative path, and `name_words` are the path's words.

*Why this order:* MiniLM truncates at 256 word pieces and was trained on English (settled decision 2). Whatever falls past the budget is lost, so the tail holds what is least likely to distinguish a node: long bodies, then parameter lists that restate the signature. Structured docs go after the body rather than after the prose deliberately — a parameter list repeats the signature in English, while the body prefix carries the error messages, SQL and URLs that people actually search for.

*Why not raw code:* settled decision 2. Results still return real code; only ranking uses the composed text. M5's A/B decides whether this holds up.

### 7. Truncation counts with the model's own tokenizer, on the joined text

The composer joins every section, encodes once without special tokens, and if the count exceeds `embed_max_tokens − 2` (for `[CLS]`/`[SEP]`) cuts the text at the character offset where the last in-budget token ends. Because sections are ordered by priority, cutting the joined text is the same as dropping trailing sections and cutting the last surviving one at a token boundary.

The tokenizer is loaded from the model repository's `tokenizer.json` with the `tokenizers` library, with **truncation and padding explicitly disabled**, and cached on the embedder instance.

*Why the separate tokenizer:* composing needs token counts whenever a file changes, but embedding is only needed if some composed text actually changed. Loading the tokenizer costs 0.11 s with no torch import; loading the model costs 1.94 s. A reformat, a line shift, or an edit past a long function's budget re-composes without ever loading the model.

*Why disable truncation:* measured — MiniLM's `tokenizer.json` ships with truncation at 128 tokens and padding on, which would cap every count at 128 and let over-budget texts through.

*Why count the joined text:* separators and section boundaries can merge or split tokens; counting pieces separately and summing drifts. One encode per node is cheap, and batch encoding is used per file.

### 8. `Embedder` is an interface with two lazy backends and a test double

```python
class Embedder(Protocol):
    model_name: str
    def tokenizer(self) -> Tokenizer: ...           # loads tokenizer only
    def embed(self, texts: Sequence[str]) -> list[bytes]: ...  # loads model on first call
```

`SentenceTransformerEmbedder` imports `sentence_transformers` (and therefore torch) inside its first `embed()` call and sets the model's `max_seq_length` to `embed_max_tokens`. `FastEmbedEmbedder` does the same with `fastembed`, and raises an actionable error naming the `onnx` extra when it isn't installed. Both return L2-normalized float32 vectors serialized for `vec0`. `Settings.embedding_backend` (`"sentence-transformers"` default, or `"fastembed"`) selects one.

On first model load the embedder compares the model's output dimension to `Settings.embedding_dim` and raises a typed error naming both values and the setting to change. A model that is not in the Hugging Face cache is downloaded; a download failure raises an error saying the first index needs network access once.

*Why lazy:* a no-op sync must not import torch (1.94 s), and neither must a sync whose changes compose to identical text. The embedder object is constructed eagerly and cheaply; cost is paid at first use.

*Why normalize:* `vec0` ranks by L2 distance; on unit vectors that ordering equals cosine similarity, which is what sentence-transformers models are trained for.

*Test double:* `FakeEmbedder` with a whitespace tokenizer and hash-derived unit vectors, counting its calls. Pipeline tests use it so they run in milliseconds and can assert "the model was never loaded". Backend adapters are unit-tested against stub modules; a small number of tests use the real cached model and skip when it isn't present locally.

### 9. A model change rebuilds `vectors`, not just a dimension change

`open_db` currently rebuilds `vectors` only when the stored dimension differs. It changes to rebuild when the stored model name *or* dimension differs, updating both.

*Why:* two models with the same dimension (MiniLM-L6 and MiniLM-L12 are both 384) produce vectors in unrelated spaces. Today, switching between them would silently rank old vectors against new query embeddings. Rebuilding empties `vectors`; decision 1's backlog re-embeds everything on the next sync without re-parsing.

### 10. Sync returns a report, and the report is M4's seam

`sync_repo(conn, repo_path, settings, embedder) -> SyncReport` with added, changed, removed and unchanged file paths, nodes written and deleted, refs written, texts embedded, per-file errors, and elapsed time.

*Why:* M4's resolver needs to know which files' refs are new and which node IDs disappeared (dangling edges, failed-ref retry). Returning it now means M4 adds a pass rather than re-deriving what changed.

### 11. FTS bodies are each node's own text, excluding its children

A node's `nodes_fts.body` is its byte range with the byte ranges of its child nodes cut out. A method body is indexed once, on the method — not again on its class and again on its file. A file's FTS body is therefore its imports and module-level statements; a class's is its class-level attributes; a markdown section's is its own prose, excluding subsections.

*Why:* the plan stores bodies only in FTS. Indexing each node's full range would triple-index every token in a method and make BM25 rank container nodes for every keyword. Script-style files with no symbols still have their whole content indexed, on the file node.

*Consequence:* a keyword match on a class attribute lands on the class, which is where that attribute lives.

### 12. `init` and `reindex` own rebuilding

- `indexter init [PATH]` (default `.`) resolves settings for the repository, creates its database if absent, and syncs. On an existing database it syncs and says so.
- `indexter reindex [PATH]` syncs an existing database, and fails with a pointer to `init` when there is none. `--full` deletes the database and its WAL/SHM files and indexes from scratch.
- Both catch `SchemaVersionMismatch`, delete the database, and index from scratch, printing that they did — the rebuild M1 decision 4 deferred to these commands. `RepoPathMismatch` is reported, never auto-repaired.
- Both print a one-paragraph summary from the `SyncReport` and list per-file errors; they exit non-zero only when the index could not be built, not when individual files had parse errors.

Database-file deletion moves from `remove` into a shared `db` helper so the three commands delete the same set of files.

*Why:* both commands are idempotent and safe to re-run, which is what a setup command wants. The database is a cache (M1 decision 4), so silently rebuilding an out-of-date one loses nothing.

### 13. Chunk fallback sizes move to 1000 / 100 bytes

*Why:* M2 left the lifted 250/25-character defaults for M3 to retune. At the measured ~3.4–3.7 bytes per token for code and prose, 1,000 bytes is about one 256-token embedding window, so a chunk's embedding sees most of the chunk instead of four chunks each seeing a line or two. A changed default changes the fingerprint (decision 5), so existing indexes re-chunk on their next sync.

## Risks / Trade-offs

- **Coarse mtime resolution** (1–2 s on some network and FAT filesystems) → an edit within the same second at the same size goes unnoticed. `reindex --full` is the escape hatch; local APFS/ext4 have nanosecond mtimes.
- **Composer format changes re-embed a whole repository** → about 1–2 s for this repository size; the fingerprint makes it automatic rather than a silent staleness. A pinned composer snapshot over the fixtures flags the need to bump the version.
- **A first `init` needs network access to fetch the model** → the error says so; later runs are offline. Documented in the README in M6.
- **CLI invocations pay model load (~2 s) whenever anything needs embedding** → the "well under a second" target is for in-process sync with a warm embedder, which is how search runs from M5 on. A no-op `reindex` never pays it.
- **Two processes syncing the same repository at once** (the MCP server and a CLI `reindex`) → per-file transactions make duplicate writes idempotent; the backlog insert checks for an existing vector inside its transaction and skips it, since `vec0` has no upsert. Busy timeout (M1) serializes writers.
- **Class/file nodes compose from their children's names** → renaming a method re-embeds its class and file node too. Accepted: renames are rare next to body edits, and the member list is what makes a class findable.
- **Upsert-preserved rowids are load-bearing** → a test asserts that a re-synced, unchanged node keeps its rowid and vector.
- **Structured-doc detection is heuristic across three doc conventions** → a missed block falls back to prose position, which only affects what gets truncated first.

## Migration Plan

Existing M1/M2-era databases hold schema version 1 and no nodes; M3 makes no DDL change and keeps version 1, so they open unchanged and the first sync populates them. Their stored fingerprint is absent, which counts as a mismatch and simply means every file is parsed — which it would be anyway. The model-change rebuild (decision 9) is a no-op for databases created with the default model. Changed chunk defaults only affect unregistered file types, and apply automatically via the fingerprint.

Rollback is deleting the database; nothing outside it is written.

## Open Questions

- **Long markdown sections.** The plan says "heading path + prose (windowed if long)", but the `vectors` table holds one vector per node. M3 embeds the first window only; the rest of the section remains keyword-searchable through FTS. Whether multi-window embedding earns a schema change is an M5 eval question.
- **Whether HTML/CSS/data nodes are worth embedding** (carried from M2). M3 embeds every kind the parsers emit; M5's eval decides, and the kind filter on `vectors` keeps them excludable meanwhile.
- **fastembed model-name parity.** The plan's fastembed measurement used the same model; implementation confirms that `sentence-transformers/all-MiniLM-L6-v2` resolves under fastembed and produces the same dimension, and otherwise documents the name to configure.
