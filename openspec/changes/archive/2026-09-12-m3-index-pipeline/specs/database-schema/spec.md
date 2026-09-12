## RENAMED Requirements

- FROM: `### Requirement: Embedding dimension change rebuilds only the vector table`
- TO: `### Requirement: Embedding model or dimension change rebuilds only the vector table`

## MODIFIED Requirements

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
