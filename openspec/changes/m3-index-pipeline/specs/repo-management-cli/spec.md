## ADDED Requirements

### Requirement: Initializing a repository index

`indexter init [PATH]` SHALL index the repository at `PATH` (default: the current directory), resolving settings for that repository, creating its database when absent, and synchronizing it. When the database already exists it SHALL synchronize it and state that the repository was already initialized. It SHALL print a summary of the sync and any per-file errors, and SHALL exit zero when the index was built even if some files had parse errors.

#### Scenario: First index

- **WHEN** `indexter init <repo>` is run for a repository with no database
- **THEN** a database is created at the repository's derived path, every eligible file is indexed, a summary of files, nodes, refs, and embeddings is printed, and the process exits zero

#### Scenario: Already initialized

- **WHEN** `indexter init <repo>` is run for a repository that already has a database
- **THEN** the database is synchronized rather than recreated, the output says it was already initialized, and the process exits zero

#### Scenario: Path is not a directory

- **WHEN** `indexter init` is given a path that does not exist or is not a directory
- **THEN** the process exits non-zero with a message naming the path and no database is created

#### Scenario: Files with parse errors

- **WHEN** a repository contains a file that fails to parse cleanly
- **THEN** the index is built, the file and its errors are listed in the output, and the process exits zero

### Requirement: Reindexing a repository

`indexter reindex [PATH]` SHALL synchronize the existing database of the repository at `PATH` (default: the current directory) and print a summary of the sync. When no database exists for the repository it SHALL exit non-zero with a message directing the user to `indexter init`. With `--full` it SHALL delete the database and its write-ahead log and shared-memory sidecars, then index the repository from scratch.

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

### Requirement: Out-of-date databases are rebuilt by the indexing commands

When `init` or `reindex` opens a database whose stored schema version does not match the current one, the command SHALL delete that database and its sidecars, index the repository from scratch, and state that it rebuilt the database. A database whose stored repository path does not match SHALL be reported as an error and left untouched.

#### Scenario: Schema version mismatch triggers a rebuild

- **WHEN** `indexter reindex <repo>` is run against a database with a different stored schema version
- **THEN** the database is rebuilt at the current schema version, the output says it was rebuilt, and the process exits zero

#### Scenario: Repository path mismatch is not repaired

- **WHEN** `indexter init <repo>` finds a database at the derived path whose stored repository path differs
- **THEN** the process exits non-zero with a message showing both paths and the database is unchanged
