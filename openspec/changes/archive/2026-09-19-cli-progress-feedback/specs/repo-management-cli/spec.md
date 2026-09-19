## MODIFIED Requirements

### Requirement: Initializing a repository index

`indexter init [PATH]` SHALL index the repository at `PATH` (default: the current directory), resolving settings for that repository, creating its database when absent, and synchronizing it. When the database already exists it SHALL synchronize it and state that the repository was already initialized. It SHALL print a summary of the sync — including, when resolution ran, edge counts by kind and call-reference outcomes by status and confidence — and any per-file errors, and SHALL exit zero when the index was built even if some files had parse errors. The summary and per-file errors SHALL be written to stdout. While indexing, it SHALL narrate its progress on stderr, covering the embedding model's preparation, the file indexing pass, graph resolution, and the embedding pass. It SHALL accept `--quiet` and `--progress` to disable and enable that narration.

#### Scenario: First index

- **WHEN** `indexter init <repo>` is run for a repository with no database
- **THEN** a database is created at the repository's derived path, every eligible file is indexed and resolved, a summary of files, nodes, refs, embeddings, edges, and call-reference outcomes is printed, and the process exits zero

#### Scenario: Already initialized

- **WHEN** `indexter init <repo>` is run for a repository that already has a database
- **THEN** the database is synchronized rather than recreated, the output says it was already initialized, and the process exits zero

#### Scenario: Path is not a directory

- **WHEN** `indexter init` is given a path that does not exist or is not a directory
- **THEN** the process exits non-zero with a message naming the path and no database is created

#### Scenario: Files with parse errors

- **WHEN** a repository contains a file that fails to parse cleanly
- **THEN** the index is built, the file and its errors are listed in the output, and the process exits zero

#### Scenario: Nothing to resolve

- **WHEN** `indexter init <repo>` is run on an already initialized repository with no changes
- **THEN** the summary omits resolution results rather than printing zeros

#### Scenario: Progress is narrated while indexing

- **WHEN** `indexter init <repo>` is run on a repository large enough that its phases exceed the painting threshold, with narration enabled
- **THEN** stderr shows the embedding pass as a proportional bar with completed and total counts, and each finished phase resolves to a completion line

#### Scenario: Narration does not disturb the summary

- **WHEN** `indexter init <repo>` is run with narration enabled and with narration disabled
- **THEN** stdout is identical in both runs

### Requirement: Reindexing a repository

`indexter reindex [PATH]` SHALL synchronize the existing database of the repository at `PATH` (default: the current directory) and print a summary of the sync. When no database exists for the repository it SHALL exit non-zero with a message directing the user to `indexter init`. With `--full` it SHALL delete the database and its write-ahead log and shared-memory sidecars, then index the repository from scratch. The summary SHALL be written to stdout, and progress SHALL be narrated on stderr under the same rules as `init`, which means a reindex with little or no work to do narrates nothing. It SHALL accept `--quiet` and `--progress`.

#### Scenario: Incremental reindex

- **WHEN** `indexter reindex <repo>` is run after one file changed
- **THEN** only that file is re-parsed, the summary reports it as changed, and the process exits zero

#### Scenario: Reindex with nothing changed

- **WHEN** `indexter reindex <repo>` is run immediately after a completed index
- **THEN** the summary reports every file unchanged and zero texts embedded

#### Scenario: Reindex without an index

- **WHEN** `indexter reindex <repo>` is run for a repository with no database
- **THEN** the process exits non-zero with a message that suggests `indexter init`

#### Scenario: Full reindex

- **WHEN** `indexter reindex --full <repo>` is run
- **THEN** the previous database and its sidecars are deleted and a new database is built with every eligible file indexed

#### Scenario: A no-op reindex narrates nothing

- **WHEN** `indexter reindex <repo>` is run with narration enabled immediately after a completed index
- **THEN** no phase is painted on stderr, because no phase exceeds the painting threshold, and the summary is still written to stdout

#### Scenario: A full reindex narrates like an init

- **WHEN** `indexter reindex --full <repo>` is run with narration enabled
- **THEN** stderr narrates the indexing and embedding passes as `init` does
