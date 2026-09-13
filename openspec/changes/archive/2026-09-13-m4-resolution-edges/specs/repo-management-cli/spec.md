## MODIFIED Requirements

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
