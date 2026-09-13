# index-sync Specification

## Purpose

Settled decision 5 requires syncing a repository before every search, so keeping a database
current has to be cheap when nothing changed and fast when one file changed — not a batch job
run occasionally. Composition and embedding on their own don't decide what needs re-parsing,
what needs re-embedding, or what to do with files that vanished.

This capability owns synchronizing a repository into its database: three-step change detection
(size and mtime, then content hash, then parse), per-file atomic writes of `files`, `nodes`,
`refs`, and `nodes_fts`, removal of vanished files and everything derived from them, a separate
embedding backlog pass that reuses vectors whose embed hash is unchanged, an index fingerprint
that forces a re-parse when parsing or composition settings change, and a sync report that
downstream milestones consume to scope their own work.

## Requirements

### Requirement: Changed files are detected in three steps

Synchronizing a repository SHALL walk it and compare each walked file against the stored `files` table. A file whose stored size and mtime both match SHALL be treated as unchanged without reading it. Otherwise the file SHALL be read and hashed; if the hash matches the stored one, only the stored size and mtime SHALL be updated. Only new files and files whose hash changed SHALL be parsed. The size and mtime stored SHALL be those observed by the walk.

#### Scenario: Unchanged file is not read

- **WHEN** a file's size and mtime match its stored row
- **THEN** its contents are not read and nothing about it is written

#### Scenario: Touched file is not re-parsed

- **WHEN** a file's mtime changed but its contents did not
- **THEN** it is read and hashed, its stored mtime is updated, and it is not parsed

#### Scenario: Edited file is re-parsed

- **WHEN** a file's contents changed
- **THEN** it is parsed and its nodes, refs, and full-text rows are replaced

#### Scenario: New file is indexed

- **WHEN** a walked file has no stored row
- **THEN** it is parsed and written, and a `files` row is created with its hash, language, size, mtime, index time, node count, and errors

#### Scenario: Undecodable file is recorded once

- **WHEN** a walked file cannot be decoded
- **THEN** it is recorded with no nodes and an error, and is not read again on later syncs until its size or mtime changes

### Requirement: Vanished files are removed

A stored file that is absent from the walk — deleted, moved, newly ignored, or newly excluded by a filter — SHALL be removed together with its nodes, the refs originating from those nodes, their full-text rows, and their vectors.

#### Scenario: Deleted file

- **WHEN** a previously indexed file is deleted and the repository is synced
- **THEN** no `files`, `nodes`, `refs`, `nodes_fts`, or `vectors` row derived from it remains

#### Scenario: Newly ignored file

- **WHEN** a pattern matching a previously indexed file is added to the repository's ignore configuration
- **THEN** the next sync removes that file's rows

#### Scenario: Renamed file

- **WHEN** a file is renamed without other changes
- **THEN** the old path's rows are removed and the new path is indexed with node IDs under the new path

### Requirement: Each file's writes are atomic

All writes derived from one file — its `files` row, node inserts, updates and deletions, ref replacement, full-text replacement, and vector invalidation — SHALL be committed in a single transaction, and so SHALL the removal of one vanished file. A file SHALL never be recorded at a content hash whose nodes are not present.

#### Scenario: Failure mid-file leaves the previous state

- **WHEN** writing one file's rows fails partway through
- **THEN** that file's stored hash, nodes, refs, and full-text rows are exactly as they were before the sync

#### Scenario: Interrupted first index keeps finished files

- **WHEN** a first index is interrupted after some files were written
- **THEN** those files are fully indexed, and the next sync processes only the remaining files

### Requirement: Nodes are written by stable identity

Writing a parsed file SHALL upsert each node on its stable text ID, preserving the stored row's integer rowid, and SHALL delete that file's stored nodes whose IDs are no longer produced. Nodes SHALL be stored with every parsed field plus the composed `qualified_name`, `name_words`, `embed_text`, `embed_hash`, and an update time; no node body SHALL be stored in `nodes`. The file's refs SHALL be replaced with the newly extracted refs, with status `unresolved`. Each node SHALL have exactly one full-text row, keyed by its rowid, holding its ID, name, name words, qualified name, docstring, signature, and full-text body.

#### Scenario: Unchanged node keeps its rowid

- **WHEN** a file is re-synced after an edit to one function
- **THEN** every other node in the file keeps the rowid it had before

#### Scenario: Removed symbol is deleted

- **WHEN** a function is deleted from a file and the file is re-synced
- **THEN** that function's node, full-text row, vector, and the refs originating from it are gone

#### Scenario: Refs are replaced, not accumulated

- **WHEN** a file is re-synced twice with the same contents after an edit
- **THEN** the refs originating from its nodes equal the refs extracted from its current contents, with no duplicates

#### Scenario: Full-text rows track nodes

- **WHEN** any sync completes
- **THEN** the set of `nodes_fts` rowids equals the set of `nodes` rowids

### Requirement: Resolution runs after structural changes

Every structural write — writing a parsed file, removing a vanished file, or recording an unreadable file — SHALL mark resolution as pending in `project_metadata` within the same transaction. After all structural writes and before the embedding backlog, the sync SHALL resolve the whole repository when resolution is pending or when the stored resolver version differs from the current one, and SHALL clear the pending mark and store the current resolver version in the same transaction as resolution's writes. A sync that makes no structural write and finds neither condition SHALL NOT run resolution.

#### Scenario: Edit triggers resolution

- **WHEN** one file's contents change and the repository is synced
- **THEN** resolution runs and the pending mark is clear afterwards

#### Scenario: Interrupted resolution heals

- **WHEN** a sync is interrupted after writing a changed file but before resolution completed, and the repository is synced again with no further changes
- **THEN** the second sync runs resolution, and no reference remains `unresolved`

#### Scenario: Resolver change re-resolves without re-parsing

- **WHEN** the resolver version changes and the repository is synced with no file changes
- **THEN** resolution runs, no file is parsed, and no text is embedded

#### Scenario: Touch alone does not trigger resolution

- **WHEN** a file's mtime changes without a content change and nothing is pending
- **THEN** resolution does not run

### Requirement: Embeddings are reused and backfilled

A node's vector SHALL be kept when a re-sync leaves its `embed_hash` unchanged and SHALL be deleted when the `embed_hash` changes. After all file writes and resolution, the sync SHALL embed every node that has no vector, other than external module nodes, in batches, inserting each vector keyed by the node's rowid together with its kind and language. Inserting a vector for a node that already has one SHALL be skipped rather than failing. The embedding model SHALL NOT be loaded when no such node lacks a vector.

#### Scenario: Unchanged text is not re-embedded

- **WHEN** a file is edited in a way that changes one function's composed text
- **THEN** exactly that function's vector (and those of any containers whose composed text changed) is recomputed, and all other vectors in the file are untouched

#### Scenario: Line shifts re-embed nothing

- **WHEN** lines are inserted at the top of a file without changing any symbol
- **THEN** the file is re-parsed and no text is embedded

#### Scenario: Interrupted embedding resumes

- **WHEN** a sync is interrupted after file writes but before all vectors were inserted
- **THEN** the next sync embeds exactly the nodes still lacking vectors, without re-parsing any file

#### Scenario: Vector table rebuilt

- **WHEN** the database's vector table was emptied by a model or dimension change
- **THEN** the next sync embeds every node without re-parsing any file

#### Scenario: Concurrent embedding

- **WHEN** a vector is inserted for a node that another process already embedded
- **THEN** the existing vector is kept and the sync continues

#### Scenario: External modules never load the model

- **WHEN** a sync's only new nodes are external module nodes
- **THEN** no text is embedded and the embedding model is not loaded

### Requirement: Format and setting changes force a re-parse

The sync SHALL store an index fingerprint in `project_metadata` derived from the indexer's format version, `chunk_size`, `chunk_overlap`, `embed_max_tokens`, and `embedding_model`, written after a sync completes. When the stored fingerprint is absent or differs from the current one, every walked file SHALL be treated as changed. Nodes whose composed text is unchanged SHALL still keep their vectors.

#### Scenario: Changed chunk size re-chunks

- **WHEN** `chunk_size` is changed and the repository is synced
- **THEN** every file is re-parsed and chunked files carry chunks of the new size

#### Scenario: Format bump without text change

- **WHEN** the format version changes but composition produces identical text for a node
- **THEN** that node's vector is kept

#### Scenario: Fingerprint written after sync

- **WHEN** a sync completes
- **THEN** `project_metadata.index_fingerprint` holds the current fingerprint

### Requirement: A no-op sync is cheap

When no walked file differs from its stored row, no stored file has vanished, the fingerprint matches, no resolution is pending, the stored resolver version matches, and every embeddable node has a vector, the sync SHALL read no file contents, load neither the tokenizer nor the model, run no resolution, and write no row other than metadata already equal to its stored value.

#### Scenario: Re-running a sync

- **WHEN** a repository is synced twice in a row with no changes between
- **THEN** the second sync reports every file unchanged, zero texts embedded, and no resolution run, and the embedder's tokenizer and model were never loaded

#### Scenario: Re-running after only a touch

- **WHEN** one file is touched without changing its contents
- **THEN** the sync reads that one file, parses nothing, resolves nothing, and embeds nothing

### Requirement: Sync reports what it did

A sync SHALL return a report containing the added, changed, removed, and unchanged file paths; the number of nodes written and deleted; the number of refs written; the number of texts embedded; the per-file errors from reading and parsing; the elapsed time; and, when resolution ran, its results — reference counts by kind and outcome, edge counts by kind and confidence, edges inserted and deleted, the number of external module nodes, and resolution's elapsed time. Per-file parse errors SHALL be recorded in the file's `errors` column and SHALL NOT stop the sync.

#### Scenario: Report after a mixed sync

- **WHEN** a sync adds one file, changes one, removes one, and leaves the rest unchanged
- **THEN** the report lists each path in the matching category and counts the nodes, refs, and embeddings involved, and includes resolution results

#### Scenario: Parse errors are reported, not raised

- **WHEN** one file has syntax errors
- **THEN** the sync completes, the report carries that file's errors, and its `files.errors` column records them

#### Scenario: No resolution results when resolution did not run

- **WHEN** a sync finds nothing changed and no resolution pending
- **THEN** the report carries no resolution results
