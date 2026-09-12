import sqlite3
import sys

import pytest

from indexter.config import Settings
from indexter.db import queries
from indexter.db.connection import (
    ExtensionLoadingUnsupported,
    RepoPathMismatch,
    SchemaVersionMismatch,
    SqliteVecLoadFailed,
    SqliteVecNotInstalled,
    _apply_pragmas,
    _load_sqlite_vec,
    open_db,
    read_metadata,
)
from indexter.db.tests.conftest import f32, insert_edge, insert_file, insert_node, insert_ref

ALL_TABLES = {"files", "nodes", "refs", "edges", "project_metadata", "nodes_fts", "vectors"}


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')").fetchall()
    return {row["name"] for row in rows if not row["name"].startswith("sqlite_")}


class TestCreation:
    def test_repo_required_to_create(self, db_path, settings):
        with pytest.raises(ValueError, match="repo is required"):
            with open_db(db_path, settings=settings):
                pass

    def test_new_db_has_all_tables(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            names = _table_names(conn)
        assert ALL_TABLES <= names

    def test_second_open_leaves_contents_untouched(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_node(conn, "a.py::foo#function")

        with open_db(db_path, repo=repo, settings=settings) as conn:
            rows = conn.execute("SELECT id FROM nodes").fetchall()
        assert [r["id"] for r in rows] == ["a.py::foo#function"]

    def test_failed_creation_leaves_no_file(self, db_path, repo, settings, monkeypatch):
        import indexter.db.connection as connection_mod

        def broken_schema():
            return "CREATE TABLE this is not valid sql;"

        monkeypatch.setattr(connection_mod, "_schema_sql", broken_schema)

        with pytest.raises(sqlite3.OperationalError):
            with open_db(db_path, repo=repo, settings=settings):
                pass

        assert not db_path.exists()
        # no leftover temp files either
        assert list(db_path.parent.glob("*.tmp-*")) == []


class TestRoundTrip:
    def test_core_tables_round_trip_with_nulls(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_file(conn, "a.py", errors=None)
            insert_node(conn, "a.py::Foo#class", docstring=None, parent_id=None)
            insert_ref(conn, "a.py::Foo#class", resolved_target_id=None, candidates='["x", "y"]')
            insert_edge(conn, "a.py::Foo#class", "a.py::Bar#class", line=None)

            file_row = conn.execute("SELECT * FROM files WHERE path = 'a.py'").fetchone()
            node_row = conn.execute("SELECT * FROM nodes WHERE id = 'a.py::Foo#class'").fetchone()
            ref_row = conn.execute("SELECT * FROM refs WHERE from_node_id = 'a.py::Foo#class'").fetchone()
            edge_row = conn.execute("SELECT * FROM edges WHERE source = 'a.py::Foo#class'").fetchone()

        assert file_row["errors"] is None
        assert node_row["docstring"] is None
        assert node_row["parent_id"] is None
        assert ref_row["resolved_target_id"] is None
        assert ref_row["candidates"] == '["x", "y"]'
        assert edge_row["line"] is None
        assert edge_row["target"] == "a.py::Bar#class"

    def test_node_id_uniqueness(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_node(conn, "a.py::foo#function")
            with pytest.raises(sqlite3.IntegrityError):
                insert_node(conn, "a.py::foo#function")

    def test_duplicate_edge_rejected(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_edge(conn, "a", "b", kind="calls", line=5)
            with pytest.raises(sqlite3.IntegrityError):
                insert_edge(conn, "a", "b", kind="calls", line=5)

    def test_duplicate_edge_with_null_line_rejected(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_edge(conn, "a", "b", kind="imports", line=None)
            with pytest.raises(sqlite3.IntegrityError):
                insert_edge(conn, "a", "b", kind="imports", line=None)

    def test_edges_differing_only_by_line_are_distinct(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_edge(conn, "a", "b", kind="calls", line=5)
            insert_edge(conn, "a", "b", kind="calls", line=6)
            (count,) = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        assert count == 2


class TestFullTextSearch:
    def test_search_returns_rowid_and_joins_back(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            rowid = insert_node(conn, "a.py::authenticate#function", name="authenticate")
            conn.execute(
                "INSERT INTO nodes_fts (rowid, id, name, name_words, qualified_name, docstring, signature, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rowid,
                    "a.py::authenticate#function",
                    "authenticate",
                    "authenticate",
                    "authenticate",
                    "",
                    "",
                    "checks the user's password",
                ),
            )

            hits = conn.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'password'").fetchall()
            assert [h["rowid"] for h in hits] == [rowid]

            joined = conn.execute(
                "SELECT n.id FROM nodes_fts f JOIN nodes n ON n.rowid = f.rowid WHERE f.rowid = ?", (rowid,)
            ).fetchone()
            assert joined["id"] == "a.py::authenticate#function"


class TestVectors:
    def test_knn_ascending_distance_order(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (1, 'function', 'python', ?)",
                (f32([1, 0, 0, 0]),),
            )
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (2, 'function', 'python', ?)",
                (f32([0, 1, 0, 0]),),
            )
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (3, 'function', 'python', ?)",
                (f32([0.9, 0.1, 0, 0]),),
            )
            rows = conn.execute(
                "SELECT node_rowid, distance FROM vectors WHERE emb MATCH ? AND k = 3 ORDER BY distance",
                (f32([1, 0, 0, 0]),),
            ).fetchall()
        assert [r["node_rowid"] for r in rows] == [1, 3, 2]
        assert rows[0]["distance"] <= rows[1]["distance"] <= rows[2]["distance"]

    def test_kind_and_language_filter_inside_knn(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (1, 'function', 'python', ?)",
                (f32([1, 0, 0, 0]),),
            )
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (2, 'class', 'python', ?)",
                (f32([1, 0, 0, 0.01]),),
            )
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (3, 'function', 'rust', ?)",
                (f32([1, 0, 0, 0.02]),),
            )
            rows = conn.execute(
                "SELECT node_rowid FROM vectors "
                "WHERE emb MATCH ? AND k = 5 AND kind = 'function' AND language = 'python'",
                (f32([1, 0, 0, 0]),),
            ).fetchall()
        assert [r["node_rowid"] for r in rows] == [1]

    def test_wrong_length_vector_rejected(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(
                    "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (1, 'function', 'python', ?)",
                    (f32([1, 0, 0]),),
                )


class TestMissingForeignKeys:
    def test_edge_to_unknown_node_accepted(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_edge(conn, "a.py::known#function", "a.py::not_yet_indexed#function")
            (count,) = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        assert count == 1

    def test_orphan_detection_returns_exactly_orphans(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_node(conn, "a.py::known#function")
            insert_edge(conn, "a.py::known#function", "a.py::known#function", kind="calls", line=1)  # not orphan
            insert_edge(conn, "a.py::known#function", "a.py::missing#function", kind="calls", line=2)  # orphan
            insert_ref(conn, "a.py::known#function", line=1)  # not orphan
            insert_ref(conn, "a.py::missing#function", line=2)  # orphan (from_node_id)

            orphan_edges = queries.orphaned_edges(conn)
            orphan_refs = queries.orphaned_refs(conn)

        assert {e["target"] for e in orphan_edges} == {"a.py::missing#function"}
        assert {r["from_node_id"] for r in orphan_refs} == {"a.py::missing#function"}


class TestPragmas:
    def test_pragma_values(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            (journal_mode,) = conn.execute("PRAGMA journal_mode").fetchone()
            (synchronous,) = conn.execute("PRAGMA synchronous").fetchone()
            (busy_timeout,) = conn.execute("PRAGMA busy_timeout").fetchone()
            (foreign_keys,) = conn.execute("PRAGMA foreign_keys").fetchone()
        assert journal_mode.lower() == "wal"
        assert synchronous == 1  # NORMAL
        assert busy_timeout >= 5000
        assert foreign_keys == 1

    def test_transaction_rollback_discards_rows(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            conn.execute("BEGIN IMMEDIATE")
            insert_node(conn, "a.py::foo#function")
            conn.execute("ROLLBACK")
            (count,) = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        assert count == 0

    def test_context_manager_closes_on_exception(self, db_path, repo, settings):
        conn_ref = None
        with pytest.raises(ValueError, match="boom"):
            with open_db(db_path, repo=repo, settings=settings) as conn:
                conn_ref = conn
                raise ValueError("boom")
        with pytest.raises(sqlite3.ProgrammingError):
            conn_ref.execute("SELECT 1")

    def test_wal_declined_emits_warning(self):
        class FakeRow:
            def __init__(self, value):
                self._value = value

            def fetchone(self):
                return (self._value,)

        class FakeConn:
            def execute(self, sql, *args):
                if sql.startswith("PRAGMA journal_mode"):
                    return FakeRow("delete")
                return FakeRow(None)

        with pytest.warns(UserWarning, match="journal_mode"):
            _apply_pragmas(FakeConn())


class TestSqliteVecFailures:
    def test_extension_loading_unsupported(self):
        class NoExtensionSupport:
            pass

        with pytest.raises(ExtensionLoadingUnsupported) as exc_info:
            _load_sqlite_vec(NoExtensionSupport())
        assert sys.executable in str(exc_info.value)

    def test_sqlite_vec_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sqlite_vec", None)
        conn = sqlite3.connect(":memory:")
        try:
            with pytest.raises(SqliteVecNotInstalled):
                _load_sqlite_vec(conn)
        finally:
            conn.close()

    def test_sqlite_vec_load_fails(self, monkeypatch):
        import sqlite_vec

        def broken_load(conn):
            raise sqlite3.OperationalError("simulated loader failure")

        monkeypatch.setattr(sqlite_vec, "load", broken_load)
        conn = sqlite3.connect(":memory:")
        try:
            with pytest.raises(SqliteVecLoadFailed, match="simulated loader failure"):
                _load_sqlite_vec(conn)
        finally:
            conn.close()


class TestMetadata:
    def test_written_at_creation(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings):
            pass
        metadata = read_metadata(db_path)
        assert metadata["repo_path"] == str(repo.resolve())
        assert metadata["model"] == settings.embedding_model
        assert metadata["dim"] == str(settings.embedding_dim)
        assert metadata["schema_version"] == "1"

    def test_version_mismatch_raises_and_leaves_file_untouched(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings):
            pass

        # Corrupt the stored version directly, bypassing open_db.
        raw = sqlite3.connect(str(db_path))
        raw.execute("UPDATE project_metadata SET value = '999' WHERE key = 'schema_version'")
        raw.commit()
        raw.close()

        with pytest.raises(SchemaVersionMismatch):
            with open_db(db_path, repo=repo, settings=settings):
                pass

        # Untouched: the bad version is still there, not silently fixed.
        assert read_metadata(db_path)["schema_version"] == "999"

    def test_metadata_readable_despite_version_mismatch(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings):
            pass
        raw = sqlite3.connect(str(db_path))
        raw.execute("UPDATE project_metadata SET value = '999' WHERE key = 'schema_version'")
        raw.commit()
        raw.close()

        metadata = read_metadata(db_path)
        assert metadata["schema_version"] == "999"

    def test_repo_path_mismatch_raises_with_both_paths(self, db_path, repo, settings, tmp_path):
        other_repo = tmp_path / "other-repo"
        other_repo.mkdir()

        with open_db(db_path, repo=repo, settings=settings):
            pass

        with pytest.raises(RepoPathMismatch) as exc_info:
            with open_db(db_path, repo=other_repo, settings=settings):
                pass

        message = str(exc_info.value)
        assert str(repo.resolve()) in message
        assert str(other_repo.resolve()) in message


class TestDimensionChange:
    def test_dimension_change_rebuilds_only_vectors(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_file(conn)
            insert_node(conn, "a.py::foo#function")
            insert_ref(conn, "a.py::foo#function")
            insert_edge(conn, "a.py::foo#function", "a.py::bar#function")
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (1, 'function', 'python', ?)",
                (f32([1, 0, 0, 0]),),
            )

        new_settings = Settings(embedding_dim=8)
        with open_db(db_path, repo=repo, settings=new_settings) as conn:
            (vector_count,) = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
            (node_count,) = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
            (ref_count,) = conn.execute("SELECT COUNT(*) FROM refs").fetchone()
            (edge_count,) = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
            (file_count,) = conn.execute("SELECT COUNT(*) FROM files").fetchone()
            # New dimension is in effect.
            conn.execute(
                "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (1, 'function', 'python', ?)",
                (f32([0] * 8),),
            )

        assert vector_count == 0
        assert node_count == 1
        assert ref_count == 1
        assert edge_count == 1
        assert file_count == 1
        assert read_metadata(db_path)["dim"] == "8"
