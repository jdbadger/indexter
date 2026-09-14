"""Tests for `search/neighbors.py` (design.md decision 5, tasks.md 2.7):
validation, unknown-ID suggestions, breadth-first traversal, hub/external/
ambiguous stopping, ordering/limiting/budgeting and rendering are exercised
over hand-built node/edge sets, the way `hit_context`/`expand` are tested,
since `read_neighbors` is a pure graph read that never synchronizes.
`neighbors_repo`/`neighbors` (which do synchronize) get their own, smaller
set of tests over a real repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from indexter.config import Settings
from indexter.db.connection import open_db
from indexter.index.embed import FakeEmbedder
from indexter.index.sync import sync_repo
from indexter.paths import db_path as resolve_db_path
from indexter.search.hybrid import IndexNotFound
from indexter.search.neighbors import (
    InvalidArgument,
    NodeNotFound,
    neighbors,
    neighbors_repo,
    read_neighbors,
    render,
)
from indexter.search.tests.conftest import insert_edge, insert_node


def insert_file(conn, *, path):
    conn.execute(
        "INSERT INTO files (path, content_hash, size, mtime, indexed_at) VALUES (?, 'h', 0, 0, 0)",
        (path,),
    )


class TestValidation:
    def test_invalid_direction(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        with pytest.raises(InvalidArgument) as excinfo:
            read_neighbors(graph_conn, "a", settings, direction="up")
        assert "direction" in str(excinfo.value)
        assert "up" in str(excinfo.value)

    def test_depth_out_of_range(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        with pytest.raises(InvalidArgument) as excinfo:
            read_neighbors(graph_conn, "a", settings, depth=4)
        assert "depth" in str(excinfo.value)
        assert "1-3" in str(excinfo.value)

    def test_unknown_edge_kind(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        with pytest.raises(InvalidArgument) as excinfo:
            read_neighbors(graph_conn, "a", settings, edges=["references"])
        assert "edges" in str(excinfo.value)
        assert "references" in str(excinfo.value)

    def test_limit_out_of_range(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        with pytest.raises(InvalidArgument):
            read_neighbors(graph_conn, "a", settings, limit=101)

    def test_blank_node_id(self, graph_conn, settings):
        with pytest.raises(InvalidArgument):
            read_neighbors(graph_conn, "  ", settings)

    def test_edges_as_a_single_string(self, graph_conn, settings):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="caller")
        insert_edge(graph_conn, source="caller", target="target")

        response = read_neighbors(graph_conn, "target", settings, direction="in", edges="calls")

        assert response.edges == ("calls",)
        assert [n.node_id for n in response.neighbors] == ["caller"]

    def test_invalid_argument_does_not_sync(self, repo, tmp_path, settings, embedder, monkeypatch):
        db_path = tmp_path / "data" / "sample.db"
        with open_db(db_path, repo=repo, settings=settings) as conn:

            def boom(*args, **kwargs):
                raise AssertionError("sync_repo should not run for an invalid argument")

            monkeypatch.setattr("indexter.search.neighbors.sync_repo", boom)
            with pytest.raises(InvalidArgument):
                neighbors_repo(conn, repo, "a", settings, embedder, direction="up")


class TestNodeNotFound:
    def test_renamed_kind_suggests_current_id(self, graph_conn, settings):
        insert_file(graph_conn, path="src/a.py")
        insert_node(graph_conn, id="src/a.py::helper#function", name="helper", file_path="src/a.py")

        with pytest.raises(NodeNotFound) as excinfo:
            read_neighbors(graph_conn, "src/a.py::helper#method", settings)

        assert excinfo.value.suggestions == ("src/a.py::helper#function",)
        assert "src/a.py::helper#function" in str(excinfo.value)

    def test_unknown_file_has_no_suggestions(self, graph_conn, settings):
        with pytest.raises(NodeNotFound) as excinfo:
            read_neighbors(graph_conn, "src/missing.py::helper#function", settings)
        assert excinfo.value.suggestions == ()

    def test_id_without_a_name_part_has_no_suggestions(self, graph_conn, settings):
        with pytest.raises(NodeNotFound) as excinfo:
            read_neighbors(graph_conn, "bogus", settings)
        assert excinfo.value.suggestions == ()


class TestUnindexedRepository:
    def test_neighbors_fails_without_creating_a_database(self, tmp_path, settings, embedder):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        with pytest.raises(IndexNotFound):
            neighbors(repo_dir, "a", settings, embedder)
        assert not any(tmp_path.rglob("*.db"))


class TestTraversal:
    def test_callers_only(self, graph_conn, settings):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="a")
        insert_node(graph_conn, id="z")
        insert_edge(graph_conn, source="a", target="target")
        insert_edge(graph_conn, source="z", target="target")

        response = read_neighbors(graph_conn, "target", settings, direction="in", edges=["calls"])

        assert [n.node_id for n in response.neighbors] == ["a", "z"]
        assert all(n.depth == 1 for n in response.neighbors)

    def test_both_directions(self, graph_conn, settings):
        insert_node(graph_conn, id="caller")
        insert_node(graph_conn, id="m", parent_id="cls")
        insert_node(graph_conn, id="callee")
        insert_node(graph_conn, id="cls", kind="class")
        insert_edge(graph_conn, source="caller", target="m", kind="calls")
        insert_edge(graph_conn, source="m", target="callee", kind="calls")
        insert_edge(graph_conn, source="cls", target="m", kind="contains")

        response = read_neighbors(graph_conn, "m", settings)

        assert {n.node_id for n in response.neighbors} == {"caller", "callee", "cls"}

    def test_depth_two(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        insert_node(graph_conn, id="b")
        insert_node(graph_conn, id="c")
        insert_edge(graph_conn, source="a", target="b")
        insert_edge(graph_conn, source="b", target="c")

        response = read_neighbors(graph_conn, "a", settings, direction="out", edges=["calls"], depth=2)

        by_id = {n.node_id: n for n in response.neighbors}
        assert by_id["b"].depth == 1
        assert by_id["c"].depth == 2
        assert by_id["c"].via_id == "b"

    def test_shallowest_depth_wins(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        insert_node(graph_conn, id="b")
        insert_node(graph_conn, id="c")
        insert_edge(graph_conn, source="a", target="b")
        insert_edge(graph_conn, source="a", target="c")
        insert_edge(graph_conn, source="b", target="c")

        response = read_neighbors(graph_conn, "a", settings, direction="out", edges=["calls"], depth=2)

        assert [n.node_id for n in response.neighbors].count("c") == 1
        c = next(n for n in response.neighbors if n.node_id == "c")
        assert c.depth == 1

    def test_importers_of_external_module(self, graph_conn, settings):
        insert_node(graph_conn, id="external::pydantic", kind="external_module", file_path="")
        insert_node(graph_conn, id="f1", kind="file")
        insert_node(graph_conn, id="f2", kind="file")
        insert_edge(graph_conn, source="f1", target="external::pydantic", kind="imports")
        insert_edge(graph_conn, source="f2", target="external::pydantic", kind="imports")

        response = read_neighbors(graph_conn, "external::pydantic", settings, direction="in", edges=["imports"])

        assert {n.node_id for n in response.neighbors} == {"f1", "f2"}


class TestWalkStops:
    def test_hub_listed_but_not_walked(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        insert_node(graph_conn, id="h", degree=41)
        insert_node(graph_conn, id="x")
        insert_edge(graph_conn, source="a", target="h")
        insert_edge(graph_conn, source="h", target="x")

        response = read_neighbors(graph_conn, "a", settings, direction="out", edges=["calls"], depth=2)

        ids = {n.node_id for n in response.neighbors}
        assert "h" in ids
        assert "x" not in ids

    def test_hub_as_start_node(self, graph_conn, settings):
        insert_node(graph_conn, id="h", degree=41)
        insert_node(graph_conn, id="caller")
        insert_edge(graph_conn, source="caller", target="h")

        response = read_neighbors(graph_conn, "h", settings, direction="in")

        assert [n.node_id for n in response.neighbors] == ["caller"]

    def test_ambiguous_only_node_not_walked(self, graph_conn, settings):
        insert_node(graph_conn, id="a")
        insert_node(graph_conn, id="b")
        insert_node(graph_conn, id="c")
        insert_edge(graph_conn, source="a", target="b", confidence="ambiguous")
        insert_edge(graph_conn, source="b", target="c")

        response = read_neighbors(graph_conn, "a", settings, direction="out", edges=["calls"], depth=2)

        ids = {n.node_id for n in response.neighbors}
        assert "b" in ids
        assert "c" not in ids
        b = next(n for n in response.neighbors if n.node_id == "b")
        assert b.confidence == "ambiguous"

    def test_walk_cap_reports_a_lower_bound(self, graph_conn, settings, monkeypatch):
        import indexter.search.neighbors as neighbors_module

        monkeypatch.setattr(neighbors_module, "WALK_NODE_CAP", 3)
        insert_node(graph_conn, id="target")
        for i in range(5):
            node_id = f"caller{i}"
            insert_node(graph_conn, id=node_id)
            insert_edge(graph_conn, source=node_id, target="target")

        response = read_neighbors(graph_conn, "target", settings, direction="in", limit=100)

        assert response.omitted_is_lower_bound
        assert len(response.neighbors) < 5


class TestOrderingLimitingBudgeting:
    def test_limit(self, graph_conn, settings):
        insert_node(graph_conn, id="target")
        for i in range(30):
            node_id = f"caller{i:02d}"
            insert_node(graph_conn, id=node_id)
            insert_edge(graph_conn, source=node_id, target="target")

        response = read_neighbors(graph_conn, "target", settings, direction="in", limit=20)

        assert len(response.neighbors) == 20
        assert response.omitted == 10
        assert not response.omitted_is_lower_bound

    def test_confident_edges_first(self, graph_conn, settings):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="ambiguous_caller")
        insert_node(graph_conn, id="exact_caller")
        insert_edge(graph_conn, source="ambiguous_caller", target="target", confidence="ambiguous")
        insert_edge(graph_conn, source="exact_caller", target="target", confidence="exact")

        response = read_neighbors(graph_conn, "target", settings, direction="in")

        assert response.neighbors[0].node_id == "exact_caller"

    def test_character_budget(self, graph_conn):
        tight_settings = Settings(embedding_dim=4, search_max_chars=80)
        insert_node(graph_conn, id="target")
        for i in range(5):
            node_id = f"caller_with_a_long_name_{i:02d}"
            insert_node(graph_conn, id=node_id, file_path="src/callers.py")
            insert_edge(graph_conn, source=node_id, target="target")

        response = read_neighbors(graph_conn, "target", tight_settings, direction="in")

        assert len(response.neighbors) < 5
        assert response.omitted > 0
        chunk = render(response)
        assert response.neighbors[0].node_id in chunk

    def test_deterministic_order(self, graph_conn, settings):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="a")
        insert_node(graph_conn, id="b")
        insert_edge(graph_conn, source="a", target="target")
        insert_edge(graph_conn, source="b", target="target")

        first = render(read_neighbors(graph_conn, "target", settings, direction="in"))
        second = render(read_neighbors(graph_conn, "target", settings, direction="in"))

        assert first == second


class TestRendering:
    def test_caller_item(self, graph_conn, settings):
        insert_node(
            graph_conn,
            id="src/w.py::Walker._should_skip#method",
            qualified_name="Walker._should_skip",
            kind="method",
            file_path="src/w.py",
            start_line=120,
            end_line=158,
        )
        insert_node(
            graph_conn,
            id="src/w.py::Walker.walk#method",
            qualified_name="Walker.walk",
            kind="method",
            file_path="src/w.py",
            start_line=170,
            end_line=230,
        )
        insert_edge(
            graph_conn,
            source="src/w.py::Walker.walk#method",
            target="src/w.py::Walker._should_skip#method",
            kind="calls",
            confidence="exact",
            line=188,
        )

        response = read_neighbors(graph_conn, "src/w.py::Walker._should_skip#method", settings, direction="in")
        text = render(response)

        assert "called by Walker.walk" in text
        assert "method" in text
        assert "src/w.py:170-230" in text
        assert "exact, line 188" in text
        assert "id: src/w.py::Walker.walk#method" in text

    def test_via_at_depth_two(self, graph_conn, settings):
        insert_node(graph_conn, id="a", qualified_name="a")
        insert_node(graph_conn, id="b", qualified_name="b")
        insert_node(graph_conn, id="c", qualified_name="c")
        insert_edge(graph_conn, source="a", target="b")
        insert_edge(graph_conn, source="b", target="c")

        response = read_neighbors(graph_conn, "a", settings, direction="out", edges=["calls"], depth=2)
        text = render(response)

        assert "via b" in text

    def test_no_neighbors(self, graph_conn, settings):
        insert_node(graph_conn, id="a")

        response = read_neighbors(graph_conn, "a", settings)
        text = render(response)

        assert "no neighbors found" in text
        assert "neighbors of a" in text


# --- Entry points: `neighbors_repo`/`neighbors` over a real, synced repo -------


class TestEntryPoints:
    def test_new_call_found_after_an_edit(self, repo, conn, repo_node_id, settings, embedder):
        (repo / "src" / "walker.py").write_text(
            (repo / "src" / "walker.py").read_text() + "\n\ndef caller():\n    return helper()\n"
        )

        helper_id = repo_node_id(conn, "src/walker.py", "helper")
        response = neighbors_repo(conn, repo, helper_id, settings, embedder, direction="in")

        assert any(n.qualified_name == "caller" for n in response.neighbors)

    def test_neighbors_opens_and_reads(self, repo, settings, embedder):
        # `repo` alone has never been synced; `neighbors` must sync it itself.
        with open_db(resolve_db_path(repo), repo=repo, settings=settings) as conn:
            sync_repo(conn, repo, settings, embedder)
            helper_id = next(
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM nodes WHERE file_path = ? AND name = ?", ("src/walker.py", "helper")
                )
            )
        response = neighbors(repo, helper_id, settings, embedder, direction="in")
        assert response.node_id == helper_id


@pytest.fixture
def repo_node_id():
    def find(conn, path, name, kind=None):
        query = "SELECT id FROM nodes WHERE file_path = ? AND name = ?"
        params = [path, name]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        row = conn.execute(query, params).fetchone()
        assert row is not None, f"no node named {name!r} in {path!r}"
        return row["id"]

    return find


# --- Snapshot over the M4 fixture repository -----------------------------------

FIXTURE = Path(__file__).parents[2] / "index" / "tests" / "fixtures" / "graph_repo"


@pytest.fixture(scope="module")
def fixture_conn(tmp_path_factory):
    fixture_settings = Settings(embedding_dim=4, embed_max_tokens=256)
    fixture_embedder = FakeEmbedder(dim=4)
    db_path = tmp_path_factory.mktemp("neighbors_fixture_db") / "test.db"
    with open_db(db_path, repo=FIXTURE, settings=fixture_settings) as conn:
        sync_repo(conn, FIXTURE, fixture_settings, fixture_embedder)
        yield conn


def test_engine_class_snapshot(fixture_conn):
    row = next(
        r
        for r in fixture_conn.execute(
            "SELECT id FROM nodes WHERE file_path = ? AND name = ?", ("python/pkg/core.py", "Engine")
        )
    )
    settings = Settings(embedding_dim=4, embed_max_tokens=256)

    response = read_neighbors(fixture_conn, row["id"], settings)

    assert render(response) == snapshot("""\
neighbors of Engine — class — python/pkg/core.py:9-11 (direction=both, edges=all, depth=1): 6 shown

id: python/pkg/core.py::Engine#class

- called by build — function — python/app.py:22-24 — imported, line 23
  id: python/app.py::build#function
- inherits from Base — class — python/pkg/core.py:4-6 — exact, line 9
  id: python/pkg/core.py::Base#class
- imported by python/app.py — file — python/app.py:1-25 — imported, line 5
  id: python/app.py::#file
- imported by python/pkg/__init__.py — file — python/pkg/__init__.py:1-4 — imported, line 3
  id: python/pkg/__init__.py::#file
- contained in python/pkg/core.py — file — python/pkg/core.py:1-23
  id: python/pkg/core.py::#file
- contains Engine.run — method — python/pkg/core.py:10-11
  id: python/pkg/core.py::Engine.run#method\
""")
