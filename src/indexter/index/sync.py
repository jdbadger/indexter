"""Per-file database writes: turn one file's parse and composition into
`files`/`nodes`/`refs`/`nodes_fts`/`vectors` rows, one transaction per file
(design.md decision 3).

Vectors are only ever deleted here, never inserted -- inserting them is pass
two's job (the embedding backlog, built in group 6). Leaving a node's vector
absent is exactly the "needs embedding" signal decision 1 relies on: a new
node never had one, and a changed node loses its stale one here.

Nodes are upserted on their stable `id` (`ON CONFLICT ... DO UPDATE`), which
keeps `rowid` -- and therefore the FTS row and vector keyed off it -- stable
across an edit that doesn't touch a given node (decision 2).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from indexter.db.connection import SchemaVersionMismatch, delete_database_files, open_db
from indexter.index.compose import INDEX_FORMAT_VERSION, compose_file
from indexter.index.graph import ResolveReport, resolution_due, resolve_repo
from indexter.parse.base import parse_file
from indexter.parse.models import Kind, ParseResult
from indexter.paths import db_path as resolve_db_path
from indexter.walk import Walker, read_file

if TYPE_CHECKING:
    from pathlib import Path

    from indexter.config import Settings
    from indexter.index.compose import ComposedNode
    from indexter.index.embed import Embedder
    from indexter.walk import WalkedFile

_FINGERPRINT_KEY = "index_fingerprint"
_RESOLUTION_PENDING_KEY = "resolution_pending"


def _mark_resolution_pending(conn: sqlite3.Connection, now: float) -> None:
    """Flag that resolution must run before the next sync completes.

    Called inside every structural write's own transaction, so an
    interruption after the write but before resolution leaves the mark set
    for the next sync to pick up (design.md decision 3).
    """
    conn.execute(
        "INSERT INTO project_metadata (key, value, updated_at) VALUES (?, '1', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = '1', updated_at = excluded.updated_at",
        (_RESOLUTION_PENDING_KEY, now),
    )


def _delete_nodes(conn: sqlite3.Connection, rowids: list[int]) -> None:
    """Delete nodes and everything keyed off their rowid: FTS rows and vectors."""
    if not rowids:
        return
    placeholders = ",".join("?" for _ in rowids)
    conn.execute(f"DELETE FROM nodes WHERE rowid IN ({placeholders})", rowids)  # noqa: S608
    conn.execute(f"DELETE FROM nodes_fts WHERE rowid IN ({placeholders})", rowids)  # noqa: S608
    conn.execute(f"DELETE FROM vectors WHERE node_rowid IN ({placeholders})", rowids)  # noqa: S608


def _delete_refs_from(conn: sqlite3.Connection, node_ids: list[str]) -> None:
    if not node_ids:
        return
    placeholders = ",".join("?" for _ in node_ids)
    conn.execute(f"DELETE FROM refs WHERE from_node_id IN ({placeholders})", node_ids)  # noqa: S608


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Counts from one `write_file` call, rolled up into a `SyncReport`."""

    nodes_written: int
    nodes_deleted: int
    refs_written: int


def write_file(
    conn: sqlite3.Connection,
    walked: WalkedFile,
    content_hash: str,
    parse_result: ParseResult,
    composed: dict[str, ComposedNode],
) -> WriteResult:
    """Write one new-or-changed file's parse in a single transaction.

    Upserts the `files` row and every node (preserving rowid for surviving
    nodes), deletes nodes whose IDs are no longer produced (with their FTS
    row, vector, and originating refs), replaces the file's refs wholesale,
    replaces every surviving/new node's FTS row, and drops the vector of any
    node whose `embed_hash` changed.
    """
    now = time.time()
    file_node = next(n for n in parse_result.nodes if n.kind == Kind.FILE)
    errors = "; ".join(parse_result.errors) if parse_result.errors else None

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO files (path, content_hash, language, size, mtime, indexed_at, node_count, errors) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "content_hash = excluded.content_hash, language = excluded.language, size = excluded.size, "
            "mtime = excluded.mtime, indexed_at = excluded.indexed_at, node_count = excluded.node_count, "
            "errors = excluded.errors",
            (
                walked.path,
                content_hash,
                file_node.language or None,
                walked.size,
                walked.mtime,
                now,
                len(parse_result.nodes),
                errors,
            ),
        )

        old_rows = conn.execute(
            "SELECT id, rowid, embed_hash FROM nodes WHERE file_path = ?", (walked.path,)
        ).fetchall()
        old_by_id = {row["id"]: (row["rowid"], row["embed_hash"]) for row in old_rows}

        _delete_refs_from(conn, list(old_by_id))

        new_ids = {node.id for node in parse_result.nodes}
        vanished_rowids = [rowid for node_id, (rowid, _) in old_by_id.items() if node_id not in new_ids]
        _delete_nodes(conn, vanished_rowids)

        for node in parse_result.nodes:
            comp = composed[node.id]
            cursor = conn.execute(
                "INSERT INTO nodes (id, kind, name, name_words, qualified_name, file_path, language, "
                "start_line, end_line, start_byte, end_byte, signature, docstring, parent_id, "
                "embed_text, embed_hash, degree, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "kind = excluded.kind, name = excluded.name, name_words = excluded.name_words, "
                "qualified_name = excluded.qualified_name, file_path = excluded.file_path, "
                "language = excluded.language, start_line = excluded.start_line, end_line = excluded.end_line, "
                "start_byte = excluded.start_byte, end_byte = excluded.end_byte, "
                "signature = excluded.signature, docstring = excluded.docstring, parent_id = excluded.parent_id, "
                "embed_text = excluded.embed_text, embed_hash = excluded.embed_hash, "
                "updated_at = excluded.updated_at "
                "RETURNING rowid",
                (
                    node.id,
                    node.kind.value,
                    node.name,
                    comp.name_words,
                    comp.qualified_name,
                    walked.path,
                    node.language or None,
                    node.start_line,
                    node.end_line,
                    node.start_byte,
                    node.end_byte,
                    node.signature,
                    node.docstring,
                    node.parent_id,
                    comp.embed_text,
                    comp.embed_hash,
                    now,
                ),
            )
            (rowid,) = cursor.fetchone()

            old_hash = old_by_id.get(node.id, (None, None))[1]
            if old_hash is not None and old_hash != comp.embed_hash:
                conn.execute("DELETE FROM vectors WHERE node_rowid = ?", (rowid,))

            conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (rowid,))
            conn.execute(
                "INSERT INTO nodes_fts (rowid, id, name, name_words, qualified_name, docstring, signature, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rowid,
                    node.id,
                    node.name,
                    comp.name_words,
                    comp.qualified_name,
                    node.docstring,
                    node.signature,
                    comp.body,
                ),
            )

        for ref in parse_result.refs:
            conn.execute(
                "INSERT INTO refs (from_node_id, raw_name, head, imported_name, for_type, ref_kind, line, col, "
                "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unresolved')",
                (
                    ref.from_node_id,
                    ref.raw_name,
                    ref.head,
                    ref.imported_name,
                    ref.for_type,
                    ref.ref_kind.value,
                    ref.line,
                    ref.col,
                ),
            )

        _mark_resolution_pending(conn, now)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")

    return WriteResult(
        nodes_written=len(parse_result.nodes),
        nodes_deleted=len(vanished_rowids),
        refs_written=len(parse_result.refs),
    )


def remove_file(conn: sqlite3.Connection, path: str) -> int:
    """Remove a file that vanished from the walk: its `files` row, nodes,
    originating refs, FTS rows, and vectors -- one transaction. Returns the
    number of nodes deleted.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute("SELECT id, rowid FROM nodes WHERE file_path = ?", (path,)).fetchall()
        _delete_refs_from(conn, [row["id"] for row in rows])
        _delete_nodes(conn, [row["rowid"] for row in rows])
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        _mark_resolution_pending(conn, time.time())
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")

    return len(rows)


def record_unreadable(conn: sqlite3.Connection, walked: WalkedFile, error: str) -> None:
    """Record a file that couldn't be decoded: empty hash, zero nodes, the
    error -- and clear any nodes/refs/FTS/vectors left from a previous,
    readable version of this file, so nothing stale lingers under its path.
    """
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute("SELECT id, rowid FROM nodes WHERE file_path = ?", (walked.path,)).fetchall()
        _delete_refs_from(conn, [row["id"] for row in rows])
        _delete_nodes(conn, [row["rowid"] for row in rows])

        conn.execute(
            "INSERT INTO files (path, content_hash, language, size, mtime, indexed_at, node_count, errors) "
            "VALUES (?, '', NULL, ?, ?, ?, 0, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "content_hash = '', language = NULL, size = excluded.size, mtime = excluded.mtime, "
            "indexed_at = excluded.indexed_at, node_count = 0, errors = excluded.errors",
            (walked.path, walked.size, walked.mtime, now, error),
        )
        _mark_resolution_pending(conn, now)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def touch_file(conn: sqlite3.Connection, walked: WalkedFile) -> None:
    """Update only `size` and `mtime` for a file whose content hash is
    unchanged -- no read, no parse, no re-composition (decision 4).
    """
    conn.execute("UPDATE files SET size = ?, mtime = ? WHERE path = ?", (walked.size, walked.mtime, walked.path))


# --- Orchestration: sync_repo, the embedding backlog, and index_repository --


def compute_fingerprint(settings: Settings) -> str:
    """Hash the settings that change what parsing/composition produce, so a
    change to any of them (or a bump of `INDEX_FORMAT_VERSION`) can force a
    full re-parse even when no file's content changed (decision 5).
    """
    payload = "|".join(
        [
            str(INDEX_FORMAT_VERSION),
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            str(settings.embed_max_tokens),
            settings.embedding_model,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_fingerprint(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM project_metadata WHERE key = ?", (_FINGERPRINT_KEY,)).fetchone()
    return row["value"] if row is not None else None


def _write_fingerprint(conn: sqlite3.Connection, fingerprint: str, now: float) -> None:
    conn.execute(
        "INSERT INTO project_metadata (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (_FINGERPRINT_KEY, fingerprint, now),
    )


@dataclass(frozen=True, slots=True)
class SyncReport:
    """What one `sync_repo` call did, and the seam M4's resolver reads
    (decision 10): which files changed shape, how much was written, and
    which files errored.
    """

    added: tuple[str, ...]
    changed: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]
    nodes_written: int
    nodes_deleted: int
    refs_written: int
    texts_embedded: int
    errors: dict[str, str]
    elapsed_seconds: float
    resolution: ResolveReport | None = None


def _embedding_backlog(conn: sqlite3.Connection, embedder: Embedder, settings: Settings) -> int:
    """Embed every node lacking a vector, in `embed_batch_size` batches, each
    its own transaction. Never calls `embedder` when there is nothing to do
    (decision 1) -- a no-op sync must not load a tokenizer or a model.
    """
    rows = conn.execute(
        "SELECT rowid, kind, language, embed_text FROM nodes "
        "WHERE rowid NOT IN (SELECT node_rowid FROM vectors) AND kind != ?",
        (Kind.EXTERNAL_MODULE.value,),
    ).fetchall()
    if not rows:
        return 0

    embedded = 0
    batch_size = settings.embed_batch_size
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embedder.embed([row["embed_text"] or "" for row in batch])

        conn.execute("BEGIN IMMEDIATE")
        try:
            for row, vector in zip(batch, vectors, strict=True):
                exists = conn.execute("SELECT 1 FROM vectors WHERE node_rowid = ?", (row["rowid"],)).fetchone()
                if exists is not None:
                    continue  # gained a vector concurrently since the selection above
                conn.execute(
                    "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (?, ?, ?, ?)",
                    (row["rowid"], row["kind"], row["language"] or "", vector),
                )
                embedded += 1
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    return embedded


def sync_repo(conn: sqlite3.Connection, repo_path: str | Path, settings: Settings, embedder: Embedder) -> SyncReport:
    """Sync a repository's database with its current files: pass one (walk,
    classify, write structural changes), resolution, then pass two (the
    embedding backlog) -- see design.md decision 1 and decision 2. Nothing
    is read, parsed, resolved, or embedded unless something on disk, or the
    resolver itself, actually looks different.
    """
    start = time.perf_counter()

    old_rows = conn.execute("SELECT path, size, mtime, content_hash FROM files").fetchall()
    old_by_path = {row["path"]: (row["size"], row["mtime"], row["content_hash"]) for row in old_rows}

    current_fingerprint = compute_fingerprint(settings)
    force_reparse = _read_fingerprint(conn) != current_fingerprint

    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    errors: dict[str, str] = {}
    nodes_written = 0
    nodes_deleted = 0
    refs_written = 0
    tokenizer = None
    seen_paths: set[str] = set()

    for walked in Walker(repo_path, settings).walk():
        seen_paths.add(walked.path)
        old = old_by_path.get(walked.path)

        if old is not None and not force_reparse and old[0] == walked.size and old[1] == walked.mtime:
            unchanged.append(walked.path)
            continue

        read_result = read_file(repo_path, walked.path)
        if read_result is None:
            error = f"cannot decode {walked.path} as utf-8 or latin-1"
            record_unreadable(conn, walked, error)
            errors[walked.path] = error
            continue

        content, content_hash = read_result

        if old is not None and not force_reparse and content_hash == old[2]:
            touch_file(conn, walked)
            unchanged.append(walked.path)
            continue

        if tokenizer is None:
            tokenizer = embedder.tokenizer()

        parse_result = parse_file(walked.path, content, settings=settings)
        composed = compose_file(walked.path, content, parse_result, tokenizer, settings.embed_max_tokens)
        result = write_file(conn, walked, content_hash, parse_result, composed)
        nodes_written += result.nodes_written
        nodes_deleted += result.nodes_deleted
        refs_written += result.refs_written
        if parse_result.errors:
            errors[walked.path] = "; ".join(parse_result.errors)

        (changed if old is not None else added).append(walked.path)

    removed: list[str] = []
    for path in old_by_path:
        if path not in seen_paths:
            removed.append(path)
            nodes_deleted += remove_file(conn, path)

    resolution = resolve_repo(conn) if resolution_due(conn) else None

    texts_embedded = _embedding_backlog(conn, embedder, settings)

    _write_fingerprint(conn, current_fingerprint, time.time())

    return SyncReport(
        added=tuple(added),
        changed=tuple(changed),
        removed=tuple(removed),
        unchanged=tuple(unchanged),
        nodes_written=nodes_written,
        nodes_deleted=nodes_deleted,
        refs_written=refs_written,
        texts_embedded=texts_embedded,
        errors=errors,
        elapsed_seconds=time.perf_counter() - start,
        resolution=resolution,
    )


DatabaseStatus = Literal["created", "rebuilt", "existing"]


@dataclass(frozen=True, slots=True)
class IndexResult:
    """What `index_repository` did to the database before syncing it."""

    db_path: Path
    status: DatabaseStatus
    report: SyncReport


def index_repository(
    repo: str | Path, settings: Settings, embedder: Embedder, *, full: bool = False
) -> IndexResult:
    """Resolve a repository's database path, create/open/rebuild it as
    needed, and run `sync_repo` -- the shared body of `indexter init` and
    `indexter reindex` (decision 12).

    A schema-version mismatch, or `full=True` on an existing database,
    deletes and recreates it before syncing -- rebuild, not migrate (M1
    decision 4). `RepoPathMismatch` is never caught: it must be reported,
    not silently repaired.
    """
    path = resolve_db_path(repo)
    existed = path.exists()

    status: DatabaseStatus
    if full and existed:
        delete_database_files(path)
        status = "rebuilt"
    elif existed:
        status = "existing"
    else:
        status = "created"

    try:
        with open_db(path, repo=repo, settings=settings) as conn:
            report = sync_repo(conn, repo, settings, embedder)
    except SchemaVersionMismatch:
        delete_database_files(path)
        status = "rebuilt"
        with open_db(path, repo=repo, settings=settings) as conn:
            report = sync_repo(conn, repo, settings, embedder)

    return IndexResult(db_path=path, status=status, report=report)
