"""Tests for `index/sync.py`'s orchestration layer: `sync_repo`'s two-pass
walk/write/embed cycle, its healing behavior across interruptions and
configuration changes, and `index_repository`'s database lifecycle.
"""

from __future__ import annotations

import struct
import time

import pytest

from indexter.db.connection import RepoPathMismatch, open_db
from indexter.index.embed import FakeEmbedder
from indexter.index.sync import _embedding_backlog, index_repository, sync_repo
from indexter.index.tests.conftest import insert_vector, sync_source

SRC_A = """\
def helper():
    return 1


def main():
    return helper()
"""

SRC_A_EDITED_BODY = """\
def helper():
    return 2


def main():
    return helper()
"""

SRC_A_LINE_INSERTED = """\
def helper():
    return 1


def main():

    return helper()
"""

SRC_B = """\
def other():
    return 2
"""


def write(repo, relpath, content):
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def set_mtime(repo, relpath, mtime):
    (repo / relpath).touch()
    import os

    os.utime(repo / relpath, (mtime, mtime))


@pytest.fixture
def embedder(settings):
    return FakeEmbedder(dim=settings.embedding_dim)


class TestSyncRepo:
    def test_first_index_reports_added_and_embeds_everything(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        write(repo, "b.py", SRC_B)

        with open_db(db_path, repo=repo, settings=settings) as conn:
            report = sync_repo(conn, repo, settings, embedder)
            total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

        assert set(report.added) == {"a.py", "b.py"}
        assert report.changed == ()
        assert report.removed == ()
        assert report.unchanged == ()
        assert report.nodes_written == total_nodes
        assert report.texts_embedded == total_nodes
        assert report.errors == {}
        assert embedder.tokenizer_loads == 1
        assert embedder.model_loads == 1

    def test_resync_unchanged_reports_all_unchanged_and_no_loads(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            report = sync_repo(conn, repo, settings, embedder)

        assert report.added == ()
        assert report.changed == ()
        assert report.removed == ()
        assert report.unchanged == ("a.py",)
        assert report.nodes_written == 0
        assert report.nodes_deleted == 0
        assert report.texts_embedded == 0
        assert embedder.tokenizer_loads == 1
        assert embedder.model_loads == 1

    def test_touch_reads_but_does_not_parse(self, repo, db_path, settings, embedder, monkeypatch):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            before = conn.execute("SELECT rowid FROM nodes ORDER BY rowid").fetchall()

            set_mtime(repo, "a.py", (repo / "a.py").stat().st_mtime + 1000)

            import indexter.index.sync as sync_mod

            read_calls = []
            original_read_file = sync_mod.read_file

            def counting_read_file(*args, **kwargs):
                read_calls.append(args)
                return original_read_file(*args, **kwargs)

            parse_calls = []
            original_parse_file = sync_mod.parse_file

            def counting_parse_file(*args, **kwargs):
                parse_calls.append(args)
                return original_parse_file(*args, **kwargs)

            monkeypatch.setattr(sync_mod, "read_file", counting_read_file)
            monkeypatch.setattr(sync_mod, "parse_file", counting_parse_file)

            report = sync_repo(conn, repo, settings, embedder)
            after = conn.execute("SELECT rowid FROM nodes ORDER BY rowid").fetchall()

        assert len(read_calls) == 1
        assert parse_calls == []
        assert report.unchanged == ("a.py",)
        assert after == before

    def test_edit_reparses_only_changed_file_and_embeds_only_changed_node(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        write(repo, "b.py", SRC_B)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            b_before = conn.execute("SELECT rowid, embed_hash FROM nodes WHERE file_path = 'b.py'").fetchall()

            write(repo, "a.py", SRC_A_EDITED_BODY)
            report = sync_repo(conn, repo, settings, embedder)

            b_after = conn.execute("SELECT rowid, embed_hash FROM nodes WHERE file_path = 'b.py'").fetchall()

        assert report.changed == ("a.py",)
        assert report.added == ()
        assert [dict(r) for r in b_after] == [dict(r) for r in b_before]
        # Only `helper`'s embed_text changed (its body); `main` and the file
        # node's composed text (member listing) are untouched.
        assert report.texts_embedded == 1

    def test_line_insertion_embeds_nothing(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            write(repo, "a.py", SRC_A_LINE_INSERTED)
            report = sync_repo(conn, repo, settings, embedder)

        assert report.changed == ("a.py",)
        assert report.texts_embedded == 0

    def test_delete_removes_file(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            (repo / "a.py").unlink()
            report = sync_repo(conn, repo, settings, embedder)

            remaining = conn.execute("SELECT 1 FROM files WHERE path = 'a.py'").fetchone()

        assert report.removed == ("a.py",)
        assert report.nodes_deleted > 0
        assert remaining is None

    def test_rename_removes_old_adds_new(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            (repo / "a.py").rename(repo / "renamed.py")
            report = sync_repo(conn, repo, settings, embedder)

        assert report.removed == ("a.py",)
        assert report.added == ("renamed.py",)

    def test_newly_ignored_file_is_removed(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            ignored_settings = settings.model_copy(update={"ignore_patterns": ("a.py",)})
            report = sync_repo(conn, repo, ignored_settings, embedder)

        assert report.removed == ("a.py",)

    def test_undecodable_file_recorded_once_and_not_reread(self, repo, db_path, settings, embedder, monkeypatch):
        # latin-1 (walk.read_file's fallback) decodes any byte sequence, so a
        # real "can't decode" case can only come from an OSError; simulate it
        # by forcing read_file to fail for this one path.
        write(repo, "bad.py", "placeholder")

        import indexter.index.sync as sync_mod

        original_read_file = sync_mod.read_file
        read_calls = []

        def failing_read_file(repo_path, relpath):
            read_calls.append(relpath)
            if relpath == "bad.py":
                return None
            return original_read_file(repo_path, relpath)

        monkeypatch.setattr(sync_mod, "read_file", failing_read_file)

        with open_db(db_path, repo=repo, settings=settings) as conn:
            report = sync_repo(conn, repo, settings, embedder)
            assert report.errors and "bad.py" in report.errors

            file_row = conn.execute("SELECT * FROM files WHERE path = 'bad.py'").fetchone()
            assert file_row["content_hash"] == ""
            assert file_row["node_count"] == 0

            read_calls.clear()
            report2 = sync_repo(conn, repo, settings, embedder)

        assert read_calls == []
        assert report2.errors == {}
        assert report2.unchanged == ("bad.py",)


class TestHealing:
    def test_backlog_resumes_after_interruption_without_reparsing(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

            # Simulate a process killed after pass one but before pass two.
            conn.execute("DELETE FROM vectors")

            import indexter.index.sync as sync_mod

            parse_calls = []
            monkeypatch_target = sync_mod.parse_file

            def counting_parse_file(*args, **kwargs):
                parse_calls.append(args)
                return monkeypatch_target(*args, **kwargs)

            sync_mod.parse_file = counting_parse_file
            try:
                report = sync_repo(conn, repo, settings, embedder)
            finally:
                sync_mod.parse_file = monkeypatch_target

        assert parse_calls == []
        assert report.unchanged == ("a.py",)
        assert report.texts_embedded == node_count

    def test_dimension_change_reembeds_without_reparsing(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

        new_settings = settings.model_copy(update={"embedding_dim": settings.embedding_dim + 1})
        new_embedder = FakeEmbedder(dim=new_settings.embedding_dim)

        import indexter.index.sync as sync_mod

        parse_calls = []
        original_parse_file = sync_mod.parse_file

        def counting_parse_file(*args, **kwargs):
            parse_calls.append(args)
            return original_parse_file(*args, **kwargs)

        sync_mod.parse_file = counting_parse_file
        try:
            with open_db(db_path, repo=repo, settings=new_settings) as conn:
                report = sync_repo(conn, repo, new_settings, new_embedder)
        finally:
            sync_mod.parse_file = original_parse_file

        assert parse_calls == []
        assert report.unchanged == ("a.py",)
        assert report.texts_embedded == node_count

    def test_model_name_change_forces_reparse_and_reembed(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

        new_settings = settings.model_copy(update={"embedding_model": settings.embedding_model + "-v2"})
        new_embedder = FakeEmbedder(dim=new_settings.embedding_dim, model_name=new_settings.embedding_model)

        with open_db(db_path, repo=repo, settings=new_settings) as conn:
            report = sync_repo(conn, repo, new_settings, new_embedder)

        assert report.changed == ("a.py",)
        assert report.nodes_written == node_count
        assert report.texts_embedded == node_count

    def test_chunk_size_change_reparses_every_file(self, repo, db_path, settings, embedder):
        write(repo, "a.txt", "word " * 500)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            new_settings = settings.model_copy(update={"chunk_size": settings.chunk_size + 1})
            report = sync_repo(conn, repo, new_settings, embedder)

        assert report.changed == ("a.txt",)

    def test_format_version_bump_with_identical_text_keeps_vectors(self, repo, db_path, settings, embedder):
        import indexter.index.sync as sync_mod

        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            before_vectors = {r["node_rowid"] for r in conn.execute("SELECT node_rowid FROM vectors")}

            monkeypatch_target = sync_mod.INDEX_FORMAT_VERSION
            sync_mod.INDEX_FORMAT_VERSION = monkeypatch_target + 1
            try:
                report = sync_repo(conn, repo, settings, embedder)
            finally:
                sync_mod.INDEX_FORMAT_VERSION = monkeypatch_target

            after_vectors = {r["node_rowid"] for r in conn.execute("SELECT node_rowid FROM vectors")}

        assert report.changed == ("a.py",)
        assert report.texts_embedded == 0
        assert after_vectors == before_vectors

    def test_parse_errors_land_in_report_and_files_errors(self, repo, db_path, settings, embedder, monkeypatch):
        import indexter.index.sync as sync_mod
        from indexter.parse.base import parse_file as real_parse_file

        write(repo, "a.py", SRC_A)

        def flaky_parse_file(relpath, content, *, settings=None):
            result = real_parse_file(relpath, content, settings=settings)
            result.errors.append("synthetic parse error")
            return result

        monkeypatch.setattr(sync_mod, "parse_file", flaky_parse_file)

        with open_db(db_path, repo=repo, settings=settings) as conn:
            report = sync_repo(conn, repo, settings, embedder)
            file_row = conn.execute("SELECT errors FROM files WHERE path = 'a.py'").fetchone()

        assert report.errors["a.py"] == "synthetic parse error"
        assert file_row["errors"] == "synthetic parse error"


class TestResolution:
    def test_no_op_sync_runs_no_resolution(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            report = sync_repo(conn, repo, settings, embedder)

        assert report.resolution is None

    def test_edit_triggers_resolution(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            write(repo, "a.py", SRC_A_EDITED_BODY)
            report = sync_repo(conn, repo, settings, embedder)

            pending = conn.execute("SELECT value FROM project_metadata WHERE key = 'resolution_pending'").fetchone()[
                "value"
            ]

        assert report.resolution is not None
        assert pending == "0"

    def test_touch_does_not_trigger_resolution(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            set_mtime(repo, "a.py", (repo / "a.py").stat().st_mtime + 1000)
            report = sync_repo(conn, repo, settings, embedder)

        assert report.resolution is None

    def test_interrupted_resolution_heals_on_next_sync(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            # Simulate a crash between pass one's commit and resolution:
            # leave the pending marker set with refs still unresolved.
            conn.execute("UPDATE project_metadata SET value = '1' WHERE key = 'resolution_pending'")
            conn.execute("UPDATE refs SET status = 'unresolved', resolved_target_id = NULL, confidence = NULL")

            report = sync_repo(conn, repo, settings, embedder)

            unresolved = conn.execute("SELECT COUNT(*) AS n FROM refs WHERE status = 'unresolved'").fetchone()["n"]

        assert report.resolution is not None
        assert unresolved == 0

    def test_resolver_version_bump_reresolves_without_reparsing_or_embedding(self, repo, db_path, settings, embedder):
        write(repo, "a.py", SRC_A)
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)

            import indexter.index.graph as graph_mod
            import indexter.index.sync as sync_mod

            original_version = graph_mod.RESOLVER_VERSION
            original_parse_file = sync_mod.parse_file
            parse_calls = []

            def counting_parse_file(*args, **kwargs):
                parse_calls.append(args)
                return original_parse_file(*args, **kwargs)

            graph_mod.RESOLVER_VERSION = original_version + 1
            sync_mod.parse_file = counting_parse_file
            try:
                report = sync_repo(conn, repo, settings, embedder)
            finally:
                sync_mod.parse_file = original_parse_file
                graph_mod.RESOLVER_VERSION = original_version

        assert report.resolution is not None
        assert parse_calls == []
        assert report.texts_embedded == 0

    def test_failed_ref_succeeds_after_definition_added(self, repo, db_path, settings, embedder):
        write(repo, "b.py", "def run():\n    return undefined_thing()\n")
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            status = conn.execute("SELECT status FROM refs").fetchone()["status"]
            assert status == "failed"

            write(repo, "a.py", "def undefined_thing():\n    return 1\n")
            sync_repo(conn, repo, settings, embedder)

            status_after = conn.execute("SELECT status FROM refs WHERE raw_name = 'undefined_thing'").fetchone()[
                "status"
            ]

        assert status_after == "resolved"


class TestIndexRepository:
    def test_create(self, repo, settings, embedder, monkeypatch, tmp_path):
        write(repo, "a.py", SRC_A)
        data_dir = tmp_path / "data"
        monkeypatch.setattr("indexter.paths.data_dir", lambda: data_dir)

        result = index_repository(repo, settings, embedder)

        assert result.status == "created"
        assert result.db_path.exists()
        assert result.report.added == ("a.py",)

    def test_existing(self, repo, settings, embedder, monkeypatch, tmp_path):
        write(repo, "a.py", SRC_A)
        data_dir = tmp_path / "data"
        monkeypatch.setattr("indexter.paths.data_dir", lambda: data_dir)

        first = index_repository(repo, settings, embedder)
        assert first.status == "created"

        second = index_repository(repo, settings, embedder)
        assert second.status == "existing"
        assert second.report.unchanged == ("a.py",)

    def test_full_rebuild(self, repo, settings, embedder, monkeypatch, tmp_path):
        write(repo, "a.py", SRC_A)
        data_dir = tmp_path / "data"
        monkeypatch.setattr("indexter.paths.data_dir", lambda: data_dir)

        index_repository(repo, settings, embedder)
        result = index_repository(repo, settings, embedder, full=True)

        assert result.status == "rebuilt"
        assert result.report.added == ("a.py",)

    def test_schema_mismatch_rebuild(self, repo, settings, embedder, monkeypatch, tmp_path):
        write(repo, "a.py", SRC_A)
        data_dir = tmp_path / "data"
        monkeypatch.setattr("indexter.paths.data_dir", lambda: data_dir)

        first = index_repository(repo, settings, embedder)
        with open_db(first.db_path, repo=repo, settings=settings) as conn:
            conn.execute("UPDATE project_metadata SET value = '999' WHERE key = 'schema_version'")

        result = index_repository(repo, settings, embedder)

        assert result.status == "rebuilt"
        assert result.report.added == ("a.py",)

    def test_version_1_database_rebuilds(self, repo, settings, embedder, monkeypatch, tmp_path):
        """A real pre-M4 database (schema_version=1) is rebuilt, not migrated."""
        write(repo, "a.py", SRC_A)
        data_dir = tmp_path / "data"
        monkeypatch.setattr("indexter.paths.data_dir", lambda: data_dir)

        first = index_repository(repo, settings, embedder)
        with open_db(first.db_path, repo=repo, settings=settings) as conn:
            conn.execute("UPDATE project_metadata SET value = '1' WHERE key = 'schema_version'")

        result = index_repository(repo, settings, embedder)

        assert result.status == "rebuilt"
        assert result.report.added == ("a.py",)

    def test_repo_path_mismatch_propagated_and_db_untouched(self, repo, settings, embedder, monkeypatch, tmp_path):
        write(repo, "a.py", SRC_A)
        data_dir = tmp_path / "data"
        monkeypatch.setattr("indexter.paths.data_dir", lambda: data_dir)

        first = index_repository(repo, settings, embedder)
        before_bytes = first.db_path.read_bytes()

        other_repo = tmp_path / "other-repo"
        other_repo.mkdir()
        write(other_repo, "a.py", SRC_A)
        monkeypatch.setattr("indexter.index.sync.resolve_db_path", lambda _repo: first.db_path)

        with pytest.raises(RepoPathMismatch):
            index_repository(other_repo, settings, embedder)

        assert first.db_path.read_bytes() == before_bytes


class TestEmbeddingBacklog:
    def test_skips_node_that_gained_a_vector_concurrently(self, repo, db_path, settings, tokenizer):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_source(conn, "a.py", SRC_A, settings, tokenizer)
            rows = conn.execute("SELECT rowid FROM nodes ORDER BY rowid").fetchall()
            target_rowid = rows[0]["rowid"]
            vec = struct.pack(f"{settings.embedding_dim}f", *([0.1] * settings.embedding_dim))

            class RacingEmbedder:
                model_name = "racing"

                def tokenizer(self):
                    return tokenizer

                def embed(self, texts):
                    # Simulate another process inserting this node's vector
                    # between our SELECT and our per-row existence check.
                    conn.execute(
                        "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (?, 'function', 'python', ?)",
                        (target_rowid, vec),
                    )
                    return [vec for _ in texts]

            embedded = _embedding_backlog(conn, RacingEmbedder(), settings)

        assert embedded == len(rows) - 1

    def test_failure_mid_batch_rolls_back(self, repo, db_path, settings, tokenizer):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_source(conn, "a.py", SRC_A, settings, tokenizer)
            vec = struct.pack(f"{settings.embedding_dim}f", *([0.1] * settings.embedding_dim))

            class ShortEmbedder:
                model_name = "short"

                def tokenizer(self):
                    return tokenizer

                def embed(self, texts):
                    return [vec] * (len(texts) - 1)  # one short -> zip(strict=True) raises

            with pytest.raises(ValueError, match="shorter"):
                _embedding_backlog(conn, ShortEmbedder(), settings)

            remaining = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

        assert remaining == 0

    def _insert_external_node(self, conn, node_id: str = "external::pydantic") -> None:
        conn.execute(
            "INSERT INTO nodes (id, kind, name, file_path, degree, updated_at) "
            "VALUES (?, 'external_module', 'pydantic', '', 0, ?)",
            (node_id, time.time()),
        )

    def test_external_module_nodes_are_never_embedded(self, repo, db_path, settings, tokenizer):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_source(conn, "a.py", SRC_A, settings, tokenizer)
            self._insert_external_node(conn)

            embedder = FakeEmbedder(dim=settings.embedding_dim)
            embedded = _embedding_backlog(conn, embedder, settings)

            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            vector_count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

        assert embedded == node_count - 1
        assert vector_count == node_count - 1

    def test_only_externals_lacking_vectors_does_not_load_model(self, repo, db_path, settings, tokenizer):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            sync_source(conn, "a.py", SRC_A, settings, tokenizer)
            for row in conn.execute("SELECT rowid FROM nodes").fetchall():
                insert_vector(conn, row["rowid"], dim=settings.embedding_dim)
            self._insert_external_node(conn)

            embedder = FakeEmbedder(dim=settings.embedding_dim)
            embedded = _embedding_backlog(conn, embedder, settings)

        assert embedded == 0
        assert embedder.model_loads == 0
