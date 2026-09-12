"""Read queries shared across the CLI and later milestones: the metadata
summary `indexter list` needs, and the orphan-detection maintenance queries
that stand in for foreign-key enforcement (see design.md decision 5).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from indexter.db.connection import read_metadata


@dataclass(frozen=True)
class RepoSummary:
    """One row of `indexter list` output, read entirely from a database's own state."""

    db_path: Path
    repo_path: str | None
    repo_exists: bool
    node_count: int
    model: str | None
    dim: str | None
    schema_version: str | None
    size_bytes: int
    indexed_at: float | None


@dataclass(frozen=True)
class CorruptDatabase:
    """A `.db` file in the data directory that isn't a readable indexter database."""

    db_path: Path
    error: str


def read_summary(db_path: Path) -> RepoSummary | CorruptDatabase:
    """Summarize a database for `indexter list`, without a full `open_db` --
    no sqlite-vec load, no schema-version check, so this works on any file,
    including ones from a different schema version.
    """
    try:
        metadata = read_metadata(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            (node_count,) = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
            (indexed_at,) = conn.execute("SELECT MAX(indexed_at) FROM files").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return CorruptDatabase(db_path=db_path, error=str(e))

    repo_path = metadata.get("repo_path")
    return RepoSummary(
        db_path=db_path,
        repo_path=repo_path,
        repo_exists=Path(repo_path).exists() if repo_path else False,
        node_count=node_count,
        model=metadata.get("model"),
        dim=metadata.get("dim"),
        schema_version=metadata.get("schema_version"),
        size_bytes=db_path.stat().st_size,
        indexed_at=indexed_at,
    )


def node_count(conn: sqlite3.Connection) -> int:
    (count,) = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
    return count


def orphaned_refs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """refs whose from_node_id, or non-null resolved_target_id, names no node."""
    return conn.execute(
        "SELECT * FROM refs "
        "WHERE from_node_id NOT IN (SELECT id FROM nodes) "
        "   OR (resolved_target_id IS NOT NULL AND resolved_target_id NOT IN (SELECT id FROM nodes))"
    ).fetchall()


def orphaned_edges(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """edges whose source or target names no node."""
    return conn.execute(
        "SELECT * FROM edges WHERE source NOT IN (SELECT id FROM nodes) OR target NOT IN (SELECT id FROM nodes)"
    ).fetchall()
