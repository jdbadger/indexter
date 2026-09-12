"""Connection lifecycle: pragmas, sqlite-vec loading, schema creation and
versioning, and project_metadata bookkeeping.

The database is a derived cache, not a system of record -- schema evolution
is rebuild, not migrate (see design.md). `open_db` either creates a brand new
database with the current schema, or opens an existing one and validates it;
it never migrates one in place.
"""

from __future__ import annotations

import importlib.resources
import platform
import sqlite3
import sys
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from indexter.config import Settings
from indexter.paths import canonical_repo_path, ensure_dir

SCHEMA_VERSION = 1


class IndexterDBError(Exception):
    """Base class for all database-layer errors."""


class ExtensionLoadingUnsupported(IndexterDBError):
    """The running interpreter's sqlite3 module cannot load extensions."""

    def __init__(self, interpreter: str) -> None:
        self.interpreter = interpreter
        super().__init__(
            f"The Python interpreter at {interpreter} does not support loading SQLite "
            "extensions, which sqlite-vec requires. Run indexter under a uv-managed "
            "Python instead, e.g. `uv run indexter ...`."
        )


class SqliteVecNotInstalled(IndexterDBError):
    """The sqlite-vec package is not importable."""

    def __init__(self) -> None:
        super().__init__(
            "The sqlite-vec package is not installed. Install it with "
            "`uv add sqlite-vec` (or `pip install sqlite-vec`)."
        )


class SqliteVecLoadFailed(IndexterDBError):
    """sqlite-vec is installed but SQLite refused to load the extension binary."""

    def __init__(self, detail: str, platform_info: str) -> None:
        self.detail = detail
        self.platform_info = platform_info
        super().__init__(f"Failed to load the sqlite-vec extension on {platform_info}: {detail}")


class SchemaVersionMismatch(IndexterDBError):
    """The database's stored schema version doesn't match this build's."""

    def __init__(self, db_path: Path, found: str | None, expected: int) -> None:
        self.db_path = db_path
        self.found = found
        self.expected = expected
        super().__init__(
            f"{db_path}: schema version {found!r} does not match expected {expected}. "
            "Re-index this repository to rebuild the database."
        )


class RepoPathMismatch(IndexterDBError):
    """The database's stored repo_path doesn't match the repo it was opened for."""

    def __init__(self, db_path: Path, stored: str | None, given: str) -> None:
        self.db_path = db_path
        self.stored = stored
        self.given = given
        super().__init__(f"{db_path}: stored repo path {stored!r} does not match {given!r}")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    (mode,) = conn.execute("PRAGMA journal_mode = WAL").fetchone()
    if mode.lower() != "wal":
        warnings.warn(
            f"journal_mode is {mode!r}, not WAL -- write-ahead logging may be "
            "unavailable on this filesystem. The database remains usable.",
            stacklevel=3,
        )


def _load_sqlite_vec(conn: sqlite3.Connection) -> str:
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError) as e:
        raise ExtensionLoadingUnsupported(interpreter=sys.executable) from e

    try:
        try:
            import sqlite_vec
        except ImportError as e:
            raise SqliteVecNotInstalled() from e

        try:
            sqlite_vec.load(conn)
        except sqlite3.OperationalError as e:
            raise SqliteVecLoadFailed(detail=str(e), platform_info=platform.platform()) from e
    finally:
        conn.enable_load_extension(False)

    (version,) = conn.execute("SELECT vec_version()").fetchone()
    return version


def _schema_sql() -> str:
    return importlib.resources.files("indexter.db").joinpath("schema.sql").read_text()


def create_vectors_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the `vectors` vec0 table sized for `dim`-dimensional embeddings.

    Kept out of schema.sql because its dimension is a runtime config value,
    not a schema constant -- see design.md decision 3.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE vectors USING vec0("
        "node_rowid INTEGER PRIMARY KEY, "
        "kind TEXT, "
        "language TEXT, "
        f"emb float[{int(dim)}])"
    )


def rebuild_vectors_table(conn: sqlite3.Connection, dim: int) -> None:
    """Drop and recreate `vectors` at a new dimension. Re-embedding, not re-parsing."""
    conn.execute("DROP TABLE IF EXISTS vectors")
    create_vectors_table(conn, dim)


def _write_metadata(conn: sqlite3.Connection, key: str, value: str, now: float) -> None:
    conn.execute(
        "INSERT INTO project_metadata(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value, now),
    )


def _read_metadata_value(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM project_metadata WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else None


def delete_database_files(db_path: Path) -> None:
    """Delete a database and its WAL/SHM sidecars. Missing sidecars are fine."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def read_metadata(db_path: Path) -> dict[str, str | None]:
    """Read project_metadata without creating a schema, loading sqlite-vec, or
    checking versions -- so `list`/`remove` work against any database file,
    including ones from a different schema version or a broken extension.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT key, value FROM project_metadata").fetchall()
        return {row["key"]: row["value"] for row in rows}
    finally:
        conn.close()


def _create_database(db_path: Path, repo: str | Path, settings: Settings) -> None:
    canonical_repo = str(canonical_repo_path(repo))
    ensure_dir(db_path.parent)
    tmp_path = db_path.with_name(f"{db_path.name}.tmp-{uuid.uuid4().hex}")

    try:
        conn = _connect(tmp_path)
        try:
            _apply_pragmas(conn)
            _load_sqlite_vec(conn)
            conn.executescript(_schema_sql())
            create_vectors_table(conn, settings.embedding_dim)
            now = time.time()
            _write_metadata(conn, "repo_path", canonical_repo, now)
            _write_metadata(conn, "model", settings.embedding_model, now)
            _write_metadata(conn, "dim", str(settings.embedding_dim), now)
            _write_metadata(conn, "schema_version", str(SCHEMA_VERSION), now)
        finally:
            conn.close()
        tmp_path.rename(db_path)
    except BaseException:
        delete_database_files(tmp_path)
        raise


def _validate_and_sync(
    conn: sqlite3.Connection, db_path: Path, repo: str | Path | None, settings: Settings
) -> None:
    stored_version = _read_metadata_value(conn, "schema_version")
    if stored_version is None or int(stored_version) != SCHEMA_VERSION:
        raise SchemaVersionMismatch(db_path=db_path, found=stored_version, expected=SCHEMA_VERSION)

    if repo is not None:
        canonical_repo = str(canonical_repo_path(repo))
        stored_repo = _read_metadata_value(conn, "repo_path")
        if stored_repo != canonical_repo:
            raise RepoPathMismatch(db_path=db_path, stored=stored_repo, given=canonical_repo)

    stored_dim = _read_metadata_value(conn, "dim")
    stored_model = _read_metadata_value(conn, "model")
    if (
        stored_dim is None
        or int(stored_dim) != settings.embedding_dim
        or stored_model != settings.embedding_model
    ):
        rebuild_vectors_table(conn, settings.embedding_dim)
        now = time.time()
        _write_metadata(conn, "dim", str(settings.embedding_dim), now)
        _write_metadata(conn, "model", settings.embedding_model, now)


@contextmanager
def open_db(
    db_path: str | Path,
    *,
    repo: str | Path | None = None,
    settings: Settings | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open (creating if needed) the database at `db_path`.

    On creation, `repo` is required -- it becomes the stored `repo_path`.
    On an existing database, `repo` (if given) is checked against the stored
    value, the schema version is checked (mismatch raises, database is left
    untouched), and a changed embedding model or dimension rebuilds only
    `vectors`.
    """
    settings = settings if settings is not None else Settings()
    db_path = Path(db_path)

    if not db_path.exists():
        if repo is None:
            raise ValueError("repo is required to create a new database")
        _create_database(db_path, repo, settings)

    conn = _connect(db_path)
    try:
        _apply_pragmas(conn)
        _load_sqlite_vec(conn)
        _validate_and_sync(conn, db_path, repo, settings)
        yield conn
    finally:
        conn.close()
