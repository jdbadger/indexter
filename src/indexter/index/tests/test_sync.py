"""Tests for `index/sync.py`: per-file writes against a real database, using
`parse_file` + `compose_file` for realistic input rather than hand-built
`ParsedNode`/`ComposedNode` fixtures.
"""

from __future__ import annotations

import pytest

from indexter.db import queries
from indexter.db.connection import open_db
from indexter.index.sync import record_unreadable, remove_file, touch_file, write_file
from indexter.index.tests.conftest import insert_vector, make_walked, node_row, sync_source

SRC_V1 = """\
def helper():
    return other()


def main():
    return helper()
"""

SRC_V1_REMOVED_HELPER = """\
def main():
    return 1
"""

SRC_V1_CHANGED_DOCSTRING = '''\
def helper():
    """Now documented."""
    return other()


def main():
    return helper()
'''


@pytest.fixture
def conn(db_path, repo, settings):
    with open_db(db_path, repo=repo, settings=settings) as conn:
        yield conn


def pending(conn) -> str | None:
    row = conn.execute("SELECT value FROM project_metadata WHERE key = 'resolution_pending'").fetchone()
    return row["value"] if row is not None else None


class TestWriteFile:
    def test_new_file_writes_every_column(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)

        file_row = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        assert file_row["content_hash"]
        assert file_row["language"] == "python"
        assert file_row["node_count"] == 3  # file, helper, main
        assert file_row["errors"] is None

        helper = node_row(conn, "a.py::helper#function")
        assert helper["kind"] == "function"
        assert helper["name"] == "helper"
        assert helper["qualified_name"] == "helper"
        assert helper["name_words"] == "helper"
        assert helper["embed_text"]
        assert helper["embed_hash"]
        assert helper["degree"] == 0
        assert helper["docstring"] is None
        assert helper["signature"] == "def helper()"

        fts_row = conn.execute("SELECT * FROM nodes_fts WHERE rowid = ?", (helper["rowid"],)).fetchone()
        assert fts_row["qualified_name"] == "helper"
        assert "return other()" in fts_row["body"]

        refs = conn.execute("SELECT * FROM refs").fetchall()
        assert {r["raw_name"] for r in refs} == {"helper", "other"}
        assert all(r["status"] == "unresolved" for r in refs)

    def test_resync_unchanged_keeps_rowid_and_vector(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        helper = node_row(conn, "a.py::helper#function")
        insert_vector(conn, helper["rowid"])

        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)

        helper_again = node_row(conn, "a.py::helper#function")
        assert helper_again["rowid"] == helper["rowid"]
        vector = conn.execute("SELECT node_rowid FROM vectors WHERE node_rowid = ?", (helper["rowid"],)).fetchone()
        assert vector is not None

    def test_changed_embed_hash_drops_vector(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        helper = node_row(conn, "a.py::helper#function")
        insert_vector(conn, helper["rowid"])

        sync_source(conn, "a.py", SRC_V1_CHANGED_DOCSTRING, settings, tokenizer)

        helper_again = node_row(conn, "a.py::helper#function")
        assert helper_again["rowid"] == helper["rowid"]
        assert helper_again["embed_hash"] != helper["embed_hash"]
        vector = conn.execute("SELECT node_rowid FROM vectors WHERE node_rowid = ?", (helper["rowid"],)).fetchone()
        assert vector is None

    def test_removed_symbol_takes_fts_vector_and_refs_with_it(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        helper = node_row(conn, "a.py::helper#function")
        insert_vector(conn, helper["rowid"])

        sync_source(conn, "a.py", SRC_V1_REMOVED_HELPER, settings, tokenizer)

        assert node_row(conn, "a.py::helper#function") is None
        assert conn.execute("SELECT 1 FROM nodes_fts WHERE rowid = ?", (helper["rowid"],)).fetchone() is None
        assert conn.execute("SELECT 1 FROM vectors WHERE node_rowid = ?", (helper["rowid"],)).fetchone() is None
        # The call ref from `main` (now calling nothing meaningful, but a ref
        # to a now-removed helper still shouldn't linger from the old parse).
        refs = conn.execute("SELECT raw_name FROM refs").fetchall()
        assert [r["raw_name"] for r in refs] == []

    def test_refs_replaced_not_accumulated(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)

        refs = conn.execute("SELECT * FROM refs").fetchall()
        assert len(refs) == 2

    def test_fts_rowids_equal_node_rowids(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)

        node_rowids = {r["rowid"] for r in conn.execute("SELECT rowid FROM nodes")}
        fts_rowids = {r["rowid"] for r in conn.execute("SELECT rowid FROM nodes_fts")}
        assert node_rowids == fts_rowids
        assert len(node_rowids) == 3

    def test_failure_mid_write_rolls_back_to_previous_state(self, conn, settings, tokenizer):
        from indexter.index.compose import compose_file
        from indexter.parse.base import parse_file
        from indexter.walk import compute_hash

        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        before_files = conn.execute("SELECT * FROM files").fetchall()
        before_nodes = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        conn.execute("UPDATE project_metadata SET value = '0' WHERE key = 'resolution_pending'")

        parse_result = parse_file("a.py", SRC_V1_CHANGED_DOCSTRING, settings=settings)
        composed = compose_file("a.py", SRC_V1_CHANGED_DOCSTRING, parse_result, tokenizer, settings.embed_max_tokens)
        # Drop one node's composed entry so the write fails partway through,
        # after some nodes have already been upserted.
        broken = dict(composed)
        del broken[parse_result.nodes[-1].id]

        walked = make_walked("a.py", SRC_V1_CHANGED_DOCSTRING)
        content_hash = compute_hash("a.py", SRC_V1_CHANGED_DOCSTRING)
        with pytest.raises(KeyError):
            write_file(conn, walked, content_hash, parse_result, broken)

        after_files = conn.execute("SELECT * FROM files").fetchall()
        after_nodes = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        assert [dict(r) for r in after_files] == [dict(r) for r in before_files]
        assert [dict(r) for r in after_nodes] == [dict(r) for r in before_nodes]
        assert pending(conn) == "0"

    def test_sets_resolution_pending(self, conn, settings, tokenizer):
        assert pending(conn) is None
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        assert pending(conn) == "1"

    def test_writes_imported_name_and_for_type(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "from collections import OrderedDict\n", settings, tokenizer)
        ref = conn.execute("SELECT * FROM refs WHERE ref_kind = 'imports'").fetchone()
        assert ref["raw_name"] == "collections"
        assert ref["imported_name"] == "OrderedDict"
        assert ref["head"] == "OrderedDict"

    def test_writes_for_type_on_rust_inherits(self, conn, settings, tokenizer):
        sync_source(conn, "a.rs", "struct Foo;\nimpl Display for Foo {}\n", settings, tokenizer)
        ref = conn.execute("SELECT * FROM refs WHERE ref_kind = 'inherits'").fetchone()
        assert ref["for_type"] == "Foo"


class TestRemoveFile:
    def test_failure_mid_remove_rolls_back(self, conn, settings, tokenizer, monkeypatch):
        import indexter.index.sync as sync_mod

        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        before = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        conn.execute("UPDATE project_metadata SET value = '0' WHERE key = 'resolution_pending'")

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(sync_mod, "_delete_nodes", boom)
        with pytest.raises(RuntimeError, match="boom"):
            remove_file(conn, "a.py")

        after = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        assert dict(after) == dict(before)
        assert pending(conn) == "0"

    def test_sets_resolution_pending(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        conn.execute("UPDATE project_metadata SET value = '0' WHERE key = 'resolution_pending'")

        remove_file(conn, "a.py")

        assert pending(conn) == "1"

    def test_leaves_nothing_behind(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        helper = node_row(conn, "a.py::helper#function")
        insert_vector(conn, helper["rowid"])

        remove_file(conn, "a.py")

        assert conn.execute("SELECT 1 FROM files WHERE path = ?", ("a.py",)).fetchone() is None
        assert conn.execute("SELECT 1 FROM nodes WHERE file_path = ?", ("a.py",)).fetchone() is None
        assert conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == 0
        assert queries.orphaned_refs(conn) == []
        assert queries.orphaned_edges(conn) == []


class TestRecordUnreadable:
    def test_stores_empty_hash_zero_nodes_and_error(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)

        walked = make_walked("a.py", "\xff\xfe")
        record_unreadable(conn, walked, "cannot decode as utf-8 or latin-1")

        file_row = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        assert file_row["content_hash"] == ""
        assert file_row["node_count"] == 0
        assert file_row["errors"] == "cannot decode as utf-8 or latin-1"
        assert conn.execute("SELECT 1 FROM nodes WHERE file_path = ?", ("a.py",)).fetchone() is None
        assert conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == 0

    def test_failure_mid_write_rolls_back(self, conn, settings, tokenizer, monkeypatch):
        import indexter.index.sync as sync_mod

        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        before = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        conn.execute("UPDATE project_metadata SET value = '0' WHERE key = 'resolution_pending'")

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(sync_mod, "_delete_nodes", boom)
        walked = make_walked("a.py", "\xff\xfe")
        with pytest.raises(RuntimeError, match="boom"):
            record_unreadable(conn, walked, "unreadable")

        after = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        assert dict(after) == dict(before)
        assert pending(conn) == "0"

    def test_new_unreadable_file(self, conn):
        walked = make_walked("bad.bin", "\xff\xfe")
        record_unreadable(conn, walked, "boom")

        file_row = conn.execute("SELECT * FROM files WHERE path = ?", ("bad.bin",)).fetchone()
        assert file_row is not None
        assert file_row["content_hash"] == ""
        assert file_row["node_count"] == 0
        assert file_row["errors"] == "boom"

    def test_sets_resolution_pending(self, conn):
        walked = make_walked("bad.bin", "\xff\xfe")
        record_unreadable(conn, walked, "boom")

        assert pending(conn) == "1"


class TestTouchFile:
    def test_updates_only_size_and_mtime(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        before = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()

        walked = make_walked("a.py", SRC_V1, mtime=before["mtime"] + 100)
        touch_file(conn, walked)

        after = conn.execute("SELECT * FROM files WHERE path = ?", ("a.py",)).fetchone()
        assert after["mtime"] == before["mtime"] + 100
        assert after["size"] == before["size"]
        assert after["content_hash"] == before["content_hash"]
        assert after["indexed_at"] == before["indexed_at"]
        assert after["node_count"] == before["node_count"]

    def test_does_not_set_resolution_pending(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", SRC_V1, settings, tokenizer)
        conn.execute("UPDATE project_metadata SET value = '0' WHERE key = 'resolution_pending'")

        walked = make_walked("a.py", SRC_V1)
        touch_file(conn, walked)

        assert pending(conn) == "0"
