from indexter.db import queries
from indexter.db.connection import open_db
from indexter.db.tests.conftest import insert_edge, insert_file, insert_node, insert_ref


class TestReadSummary:
    def test_summarizes_a_healthy_database(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_file(conn)
            insert_node(conn, "a.py::foo#function")

        summary = queries.read_summary(db_path)

        assert isinstance(summary, queries.RepoSummary)
        assert summary.repo_path == str(repo.resolve())
        assert summary.repo_exists is True
        assert summary.node_count == 1
        assert summary.model == settings.embedding_model
        assert summary.dim == str(settings.embedding_dim)
        assert summary.schema_version == "2"
        assert summary.size_bytes > 0
        assert summary.indexed_at is not None

    def test_reports_missing_repo(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings):
            pass
        repo.rmdir()

        summary = queries.read_summary(db_path)
        assert summary.repo_exists is False

    def test_reports_corrupt_database(self, tmp_path):
        bad = tmp_path / "not-a-real.db"
        bad.write_bytes(b"this is not a sqlite database")

        summary = queries.read_summary(bad)

        assert isinstance(summary, queries.CorruptDatabase)
        assert summary.db_path == bad
        assert summary.error


class TestNodeCount:
    def test_counts_nodes(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_node(conn, "a.py::foo#function")
            insert_node(conn, "a.py::bar#function")
            assert queries.node_count(conn) == 2


class TestOrphanDetection:
    def test_no_orphans_in_a_consistent_graph(self, db_path, repo, settings):
        with open_db(db_path, repo=repo, settings=settings) as conn:
            insert_node(conn, "a.py::foo#function")
            insert_node(conn, "a.py::bar#function")
            insert_edge(conn, "a.py::foo#function", "a.py::bar#function")
            insert_ref(conn, "a.py::foo#function")

            assert queries.orphaned_edges(conn) == []
            assert queries.orphaned_refs(conn) == []
