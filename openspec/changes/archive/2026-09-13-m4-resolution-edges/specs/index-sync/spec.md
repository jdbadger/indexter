## MODIFIED Requirements

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

## ADDED Requirements

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
