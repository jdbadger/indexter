# database-schema Specification

## Purpose

The rewrite replaces a Docker-hosted Qdrant deployment with a single SQLite file per repository
holding embeddings, an FTS5 keyword index, and a code graph. Nothing else can be built until that
file exists with the right shape, opens with the right pragmas, and has the sqlite-vec extension
loaded and verified — this is the foundation every later milestone writes into.

This capability owns the v1 schema (`files`, `nodes`, `refs`, `edges`, `project_metadata`,
`nodes_fts`, `vectors`), the connection lifecycle that creates that schema atomically on first
open and applies pragmas on every open, loading and verifying sqlite-vec with no fallback, and the
`project_metadata` bookkeeping that records and checks the repository path, embedding model,
dimension, and schema version.

## Requirements

### Requirement: Complete v1 schema created on first open

Opening a database that does not yet exist SHALL create it with the full v1 schema in a single transaction: the `files`, `nodes`, `refs`, `edges`, and `project_metadata` tables, the `nodes_fts` FTS5 virtual table, the `vectors` vec0 virtual table, and their supporting indexes. Creation SHALL be atomic — a failure part-way SHALL leave no partially built database behind.

#### Scenario: New database is created with every table

- **WHEN** a database is opened at a path where no file exists
- **THEN** the file is created and contains the `files`, `nodes`, `refs`, `edges`, `project_metadata`, `nodes_fts`, and `vectors` tables

#### Scenario: Existing database is not recreated

- **WHEN** a database that already contains the v1 schema is opened again
- **THEN** the existing tables and their contents are left unchanged

#### Scenario: Failed creation leaves no partial database

- **WHEN** schema creation fails part-way through
- **THEN** no database file with a partial schema remains at that path

### Requirement: Schema round-trips every table

Each table SHALL accept a representative row and return it unchanged, including JSON-valued and nullable columns, and SHALL enforce its declared uniqueness constraints.

#### Scenario: Core tables round-trip

- **WHEN** a row is inserted into `files`, `nodes`, `refs`, or `edges` and then selected back
- **THEN** every column holds the value that was written, with `NULL` preserved for columns written as `NULL`

#### Scenario: Node text IDs are unique

- **WHEN** two rows are inserted into `nodes` with the same `id`
- **THEN** the second insert fails with a uniqueness violation

#### Scenario: Duplicate edges are rejected

- **WHEN** two edges are inserted with the same source, target, kind, and line
- **THEN** the second insert fails with a uniqueness violation

#### Scenario: Edges differing only by line are distinct

- **WHEN** two edges share source, target, and kind but have different line numbers
- **THEN** both inserts succeed

#### Scenario: Full-text search returns inserted content

- **WHEN** a row is inserted into `nodes_fts` with an explicit rowid matching a node's rowid, and a term from its body is searched
- **THEN** the search returns that rowid, and the row joins back to the correct node

#### Scenario: Vector search returns nearest neighbours

- **WHEN** vectors are inserted into `vectors` and a KNN query is run against a query vector
- **THEN** results are returned in ascending distance order and each `node_rowid` corresponds to a row in `nodes`

#### Scenario: Vector metadata filters apply inside the KNN

- **WHEN** a KNN query constrains `kind` or `language` to a set of values
- **THEN** only vectors whose metadata matches are returned

### Requirement: References between tables are unconstrained by foreign keys

The schema SHALL NOT declare foreign key constraints on `refs.from_node_id`, `edges.source`, or `edges.target`. These columns hold stable text node IDs that are permitted to reference nodes that do not yet exist or no longer exist.

#### Scenario: An edge to an unknown node is accepted

- **WHEN** an edge is inserted whose target ID matches no row in `nodes`
- **THEN** the insert succeeds

#### Scenario: Orphans are detectable by query

- **WHEN** edges and refs exist whose source or target IDs match no node
- **THEN** a maintenance query returns exactly those orphaned rows

### Requirement: Connection pragmas applied on every open

Every connection SHALL be opened with write-ahead logging, `synchronous=NORMAL`, a busy timeout of at least five seconds, `temp_store=MEMORY`, and `foreign_keys=ON`, and SHALL use a row factory giving column access by name. Python SHALL NOT issue implicit transactions, so callers control transaction boundaries explicitly.

#### Scenario: Pragmas are in effect

- **WHEN** a connection is opened and the pragmas are queried back
- **THEN** journal mode is `wal`, synchronous is `NORMAL`, the busy timeout is at least 5000 ms, and foreign keys are enabled

#### Scenario: Write-ahead logging unavailable

- **WHEN** the filesystem refuses write-ahead logging and journal mode falls back to another value
- **THEN** a warning is emitted naming the actual journal mode, and the connection remains usable

#### Scenario: Callers control transactions

- **WHEN** a caller opens an explicit transaction, writes rows, and rolls back
- **THEN** none of those rows are present afterwards

#### Scenario: Connection is closed by its context manager

- **WHEN** a connection context manager exits, including by exception
- **THEN** the connection is closed

### Requirement: sqlite-vec loaded and verified on every open

Every connection SHALL load the `sqlite-vec` extension and verify it before returning. There SHALL be no fallback implementation of vector search.

#### Scenario: Extension is loaded and verified

- **WHEN** a connection is opened successfully
- **THEN** the extension version query returns a version, and extension loading is disabled again before the connection is handed to the caller

#### Scenario: Interpreter does not support extension loading

- **WHEN** the running interpreter's SQLite bindings do not support loading extensions
- **THEN** opening fails with an error naming the interpreter path and directing the user to a uv-managed Python

#### Scenario: sqlite-vec package is not installed

- **WHEN** the `sqlite-vec` package cannot be imported
- **THEN** opening fails with an error naming the package and the command to install it

#### Scenario: Extension binary fails to load

- **WHEN** the extension is present but the database engine refuses to load it
- **THEN** opening fails with an error containing the underlying loader message

### Requirement: Project metadata recorded and enforced

At creation the system SHALL record the canonical repository path, embedding model name, embedding dimension, and schema version in `project_metadata`, each with an update timestamp. On subsequent opens these values SHALL be checked against the current expectations.

#### Scenario: Metadata written at creation

- **WHEN** a database is created
- **THEN** `project_metadata` contains `repo_path`, `model`, `dim`, and `schema_version` with the values in effect at creation

#### Scenario: Schema version mismatch is reported, not repaired

- **WHEN** a database whose stored schema version differs from the current one is opened
- **THEN** opening fails with a typed error carrying the database path, the stored version, and the expected version, and the database file is left untouched

#### Scenario: Metadata can be read without a full open

- **WHEN** stored metadata is read from a database whose schema version does not match, or whose extension cannot be loaded
- **THEN** the metadata values are returned successfully

#### Scenario: Repository path mismatch is reported

- **WHEN** a database is opened for a repository whose canonical path differs from the stored `repo_path`
- **THEN** opening fails with an error showing both paths

### Requirement: Embedding model or dimension change rebuilds only the vector table

The `vectors` virtual table SHALL be created with the embedding dimension taken from configuration rather than hardcoded in the schema file. When the configured embedding model or the configured dimension differs from the stored one, the system SHALL drop and recreate the `vectors` table at the configured dimension and update both the stored model and the stored dimension, leaving `nodes`, `refs`, `edges`, `files`, and `nodes_fts` intact.

#### Scenario: Vector table uses the configured dimension

- **WHEN** a database is created with a configured embedding dimension
- **THEN** the `vectors` table accepts vectors of that dimension and rejects vectors of another length

#### Scenario: Dimension change rebuilds vectors only

- **WHEN** a database holding nodes, edges, and vectors is opened with a different configured embedding dimension
- **THEN** the `vectors` table is empty and dimensioned to the new value, the stored dimension is updated, and the row counts of `nodes`, `refs`, `edges`, and `files` are unchanged

#### Scenario: Model change at the same dimension rebuilds vectors

- **WHEN** a database holding vectors is opened with a different configured embedding model of the same dimension
- **THEN** the `vectors` table is empty, the stored model name is updated, and the row counts of `nodes`, `refs`, `edges`, and `files` are unchanged

#### Scenario: Unchanged model and dimension keep vectors

- **WHEN** a database holding vectors is opened with the same configured model and dimension
- **THEN** the `vectors` table is left untouched
