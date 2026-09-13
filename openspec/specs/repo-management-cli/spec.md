# repo-management-cli Specification

## Purpose

With no `repos.json` registry, the set of indexed repositories exists only as database files on
disk, each carrying its own metadata. Users still need to see what is indexed and remove what they
no longer want, without a central index file that could drift from reality.

This capability provides that: a single `indexter` command group as the shared entry point; the
`init` and `reindex` commands that build and refresh a repository's database, rebuilding it
automatically when its stored schema version is out of date; and the `list` and `remove` commands
that enumerate and delete database files directly, always reading state from each database's own
stored metadata rather than a registry.

## Requirements

### Requirement: Single CLI entry point

The system SHALL expose one console command, `indexter`, backed by a command group. Running it with no arguments or with `--help` SHALL list the available commands. Every command SHALL exit non-zero on failure and zero on success.

#### Scenario: Help lists the commands

- **WHEN** `indexter --help` is run
- **THEN** the output lists the available commands and the process exits zero

#### Scenario: Unknown command fails clearly

- **WHEN** a command name that does not exist is invoked
- **THEN** the process exits non-zero with a message naming the unknown command

### Requirement: Initializing a repository index

`indexter init [PATH]` SHALL index the repository at `PATH` (default: the current directory), resolving settings for that repository, creating its database when absent, and synchronizing it. When the database already exists it SHALL synchronize it and state that the repository was already initialized. It SHALL print a summary of the sync — including, when resolution ran, edge counts by kind and call-reference outcomes by status and confidence — and any per-file errors, and SHALL exit zero when the index was built even if some files had parse errors.

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

### Requirement: Listing indexed repositories

`indexter list` SHALL enumerate the database files in the data directory and report, for each, the repository path, whether that path still exists, the node count, the embedding model and dimension, the schema version, the file size, and the time it was last indexed. It SHALL read this from each database's own stored metadata, never from a registry file.

#### Scenario: Indexed repositories are listed

- **WHEN** `indexter list` is run and the data directory holds indexed repositories
- **THEN** one row per database is shown with its repository path, node count, model, schema version, size, and last-indexed time

#### Scenario: No repositories indexed

- **WHEN** `indexter list` is run and the data directory is empty or absent
- **THEN** a message states that no repositories are indexed and the process exits zero

#### Scenario: Repository has been moved or deleted

- **WHEN** a database's stored repository path no longer exists on disk
- **THEN** that row is marked as missing and the command still exits zero

#### Scenario: Database from a different schema version is still listed

- **WHEN** a database's stored schema version does not match the current one
- **THEN** it is listed with its stored version shown, rather than causing the command to fail

#### Scenario: Unreadable database file is reported

- **WHEN** a file in the data directory ending in `.db` is not a readable indexter database
- **THEN** it is listed as corrupt with its filename, the remaining databases are still listed, and the process exits zero

### Requirement: Removing an indexed repository

`indexter remove` SHALL accept either a repository path or a database filename, SHALL identify the target database by the same derivation used to create it, and SHALL delete the database file together with its write-ahead log and shared-memory sidecars. It SHALL NOT touch the repository itself.

#### Scenario: Remove by repository path

- **WHEN** `indexter remove <repo path>` is confirmed
- **THEN** the database for that repository and its `-wal` and `-shm` sidecars are deleted and the repository directory is untouched

#### Scenario: Remove by database filename

- **WHEN** `indexter remove <database filename>` is confirmed
- **THEN** that database and its sidecars are deleted

#### Scenario: Confirmation is required by default

- **WHEN** `indexter remove` is run without a confirmation flag and the prompt is declined
- **THEN** nothing is deleted and the process exits zero

#### Scenario: Confirmation can be skipped

- **WHEN** `indexter remove` is run with the confirmation flag
- **THEN** the database is deleted without prompting

#### Scenario: Removing a repository that is not indexed

- **WHEN** `indexter remove` names a repository or database with no corresponding file in the data directory
- **THEN** the process exits non-zero with a message saying nothing is indexed for that target

#### Scenario: Repository need not still exist

- **WHEN** `indexter remove` names a repository path that has been deleted from disk but whose database is present
- **THEN** the database is removed successfully
