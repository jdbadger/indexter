-- Indexter v1 schema.
--
-- The `vectors` vec0 virtual table is NOT here: its dimension depends on the
-- embedding model in effect, so it is created programmatically by
-- `db/connection.py` (see create_vectors_table). Everything else is fixed.
--
-- No foreign key constraints: `refs.from_node_id`, `edges.source`, and
-- `edges.target` hold stable text node IDs that must be allowed to dangle
-- across the two-pass resolution and per-file incremental sync. Integrity is
-- checked by query (see db/queries.py), not enforced per-statement.

CREATE TABLE files (
    path        TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    language    TEXT,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    indexed_at  REAL NOT NULL,
    node_count  INTEGER NOT NULL DEFAULT 0,
    errors      TEXT
);

CREATE TABLE nodes (
    rowid           INTEGER PRIMARY KEY,  -- unstable; vectors/FTS key off this
    id              TEXT NOT NULL UNIQUE, -- stable text ID, e.g. src/a.py::Foo.bar#method
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL,
    name_words      TEXT,
    qualified_name  TEXT,
    file_path       TEXT NOT NULL,
    language        TEXT,
    start_line      INTEGER,
    end_line        INTEGER,
    start_byte      INTEGER,
    end_byte        INTEGER,
    signature       TEXT,
    docstring       TEXT,
    parent_id       TEXT,
    embed_text      TEXT,
    embed_hash      TEXT,
    degree          INTEGER NOT NULL DEFAULT 0,
    updated_at      REAL NOT NULL
);

CREATE INDEX idx_nodes_file_path ON nodes(file_path);
CREATE INDEX idx_nodes_name ON nodes(name);
CREATE INDEX idx_nodes_kind ON nodes(kind);
CREATE INDEX idx_nodes_parent_id ON nodes(parent_id);

CREATE TABLE refs (
    id                  INTEGER PRIMARY KEY,
    from_node_id        TEXT NOT NULL,
    raw_name            TEXT NOT NULL,
    head                TEXT,
    imported_name       TEXT,
    for_type            TEXT,
    ref_kind            TEXT NOT NULL,
    line                INTEGER,
    col                 INTEGER,
    status              TEXT NOT NULL DEFAULT 'unresolved',
    resolved_target_id  TEXT,
    confidence          TEXT,
    candidates          TEXT  -- JSON array
);

CREATE INDEX idx_refs_from_node_id ON refs(from_node_id);
CREATE INDEX idx_refs_status ON refs(status);
CREATE INDEX idx_refs_head ON refs(head);

CREATE TABLE edges (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    line        INTEGER,
    confidence  TEXT NOT NULL
);

-- Table-level UNIQUE constraints can't take expressions, so IFNULL(line, -1)
-- (treating "no line" as one value, not distinct NULLs) is a unique index.
CREATE UNIQUE INDEX ux_edges_source_target_kind_line ON edges(source, target, kind, IFNULL(line, -1));
CREATE INDEX idx_edges_source_kind ON edges(source, kind);
CREATE INDEX idx_edges_target_kind ON edges(target, kind);

CREATE TABLE project_metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  REAL NOT NULL
);

-- Bodies are indexed here only -- `nodes` never stores source text, since
-- snippets are always read fresh from disk by byte range after sync.
CREATE VIRTUAL TABLE nodes_fts USING fts5(
    id UNINDEXED,
    name,
    name_words,
    qualified_name,
    docstring,
    signature,
    body
);
