## ADDED Requirements

### Requirement: Single CLI entry point

The system SHALL expose one console command, `indexter`, backed by a command group. Running it with no arguments or with `--help` SHALL list the available commands. Every command SHALL exit non-zero on failure and zero on success.

#### Scenario: Help lists the commands

- **WHEN** `indexter --help` is run
- **THEN** the output lists the available commands and the process exits zero

#### Scenario: Unknown command fails clearly

- **WHEN** a command name that does not exist is invoked
- **THEN** the process exits non-zero with a message naming the unknown command

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
