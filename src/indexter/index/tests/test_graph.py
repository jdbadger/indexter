"""Tests for `index/graph.py`: edge derivation from resolved refs, external
module node lifecycle, `nodes.degree`, and the consistency guarantees
design.md decision 1 promises after every resolution run.

Files are written through `write_file`/`remove_file` directly (via
`sync_source`, group 4's helper) rather than through `sync_repo`, so these
tests exercise `resolve_repo` in isolation from the sync-level gating that
`test_sync_repo.py` covers.
"""

from __future__ import annotations

import pytest

from indexter.db.connection import open_db
from indexter.db.queries import resolution_summary
from indexter.index import graph as graph_mod
from indexter.index.graph import resolve_repo
from indexter.index.sync import remove_file
from indexter.index.tests.conftest import sync_source


@pytest.fixture
def conn(db_path, repo, settings):
    with open_db(db_path, repo=repo, settings=settings) as conn:
        yield conn


def file_node_id(conn, path: str) -> str:
    row = conn.execute("SELECT id FROM nodes WHERE file_path = ? AND kind = 'file'", (path,)).fetchone()
    return row["id"]


def node_id(conn, path: str, name: str, kind: str | None = None) -> str:
    query = "SELECT id FROM nodes WHERE file_path = ? AND name = ?"
    params: list[str] = [path, name]
    if kind is not None:
        query += " AND kind = ?"
        params.append(kind)
    row = conn.execute(query, params).fetchone()
    assert row is not None, f"no {kind or 'node'} named {name!r} in {path!r}"
    return row["id"]


def edge_rows(conn, **where):
    query = "SELECT id, source, target, kind, line, confidence FROM edges"
    if where:
        query += " WHERE " + " AND ".join(f"{k} = ?" for k in where)
    return conn.execute(query, list(where.values())).fetchall()


def edge_set(conn, **where):
    return {(r["source"], r["target"], r["kind"], r["line"], r["confidence"]) for r in edge_rows(conn, **where)}


def degree(conn, node_id: str) -> int:
    return conn.execute("SELECT degree FROM nodes WHERE id = ?", (node_id,)).fetchone()["degree"]


def ref_row(conn, from_path: str, from_name: str):
    return conn.execute(
        "SELECT refs.* FROM refs JOIN nodes ON nodes.id = refs.from_node_id "
        "WHERE nodes.file_path = ? AND nodes.name = ?",
        (from_path, from_name),
    ).fetchone()


class TestContainsEdges:
    def test_method_contained_by_class(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "class Handler:\n    def login(self):\n        return 1\n", settings, tokenizer)
        resolve_repo(conn)

        file_id = file_node_id(conn, "a.py")
        class_id = node_id(conn, "a.py", "Handler", "class")
        method_id = node_id(conn, "a.py", "login", "method")

        assert edge_set(conn, kind="contains") == {
            (file_id, class_id, "contains", None, "exact"),
            (class_id, method_id, "contains", None, "exact"),
        }

    def test_removed_symbol_loses_containment_edge(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "class Handler:\n    def login(self):\n        return 1\n", settings, tokenizer)
        resolve_repo(conn)

        sync_source(conn, "a.py", "class Handler:\n    pass\n", settings, tokenizer)
        resolve_repo(conn)

        class_id = node_id(conn, "a.py", "Handler", "class")
        targets = {t for (_s, t, k, _l, _c) in edge_set(conn, kind="contains") if _s == class_id}
        assert targets == set()


class TestResolvedEdges:
    def test_call_edge_carries_line_and_confidence(self, conn, settings, tokenizer):
        src = "def helper():\n    return 1\n\n\ndef main():\n    return helper()\n"
        sync_source(conn, "a.py", src, settings, tokenizer)
        resolve_repo(conn)

        helper_id = node_id(conn, "a.py", "helper", "function")
        main_id = node_id(conn, "a.py", "main", "function")
        assert (main_id, helper_id, "calls", 6, "exact") in edge_set(conn, kind="calls")

    def test_ambiguous_call_fans_out(self, conn, settings, tokenizer):
        for path in ("pkg_a.py", "pkg_b.py", "pkg_c.py"):
            sync_source(conn, path, "def parse():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "caller.py", "def run():\n    return parse()\n", settings, tokenizer)
        resolve_repo(conn)

        run_id = node_id(conn, "caller.py", "run", "function")
        candidate_ids = {node_id(conn, p, "parse", "function") for p in ("pkg_a.py", "pkg_b.py", "pkg_c.py")}

        calls = edge_set(conn, kind="calls")
        assert calls == {(run_id, target, "calls", 2, "ambiguous") for target in candidate_ids}

    def test_import_of_external_package(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "import pydantic\n", settings, tokenizer)
        resolve_repo(conn)

        file_id = file_node_id(conn, "a.py")
        assert (file_id, "external::pydantic", "imports", 1, "imported") in edge_set(conn, kind="imports")

    def test_call_into_external_package_has_no_edge(self, conn, settings, tokenizer):
        src = "import pydantic\n\n\ndef use_it():\n    return pydantic.Field()\n"
        sync_source(conn, "a.py", src, settings, tokenizer)
        resolve_repo(conn)

        assert edge_set(conn, kind="calls") == set()
        ref = ref_row(conn, "a.py", "use_it")
        assert ref["status"] == "external"
        assert ref["resolved_target_id"] == "external::pydantic"

    def test_import_edge_targets_most_specific_node(self, conn, settings, tokenizer):
        sync_source(conn, "pkg/auth.py", "def login():\n    return 1\n", settings, tokenizer)
        consumer_src = "from pkg.auth import login\n\n\ndef call_it():\n    return login()\n"
        sync_source(conn, "consumer.py", consumer_src, settings, tokenizer)
        resolve_repo(conn)

        login_id = node_id(conn, "pkg/auth.py", "login", "function")
        consumer_file_id = file_node_id(conn, "consumer.py")
        assert (consumer_file_id, login_id, "imports", 1, "imported") in edge_set(conn, kind="imports")

    def test_inheritance_edge(self, conn, settings, tokenizer):
        src = "class User:\n    pass\n\n\nclass Admin(User):\n    pass\n"
        sync_source(conn, "d.py", src, settings, tokenizer)
        resolve_repo(conn)

        user_id = node_id(conn, "d.py", "User", "class")
        admin_id = node_id(conn, "d.py", "Admin", "class")
        assert (admin_id, user_id, "inherits", 5, "exact") in edge_set(conn, kind="inherits")

    def test_two_calls_on_one_line(self, conn, settings, tokenizer):
        src = "def f(x):\n    return x\n\n\ndef g():\n    return f(f(1))\n"
        sync_source(conn, "b.py", src, settings, tokenizer)
        resolve_repo(conn)

        f_id = node_id(conn, "b.py", "f", "function")
        g_id = node_id(conn, "b.py", "g", "function")
        matching = [e for e in edge_rows(conn, kind="calls") if e["source"] == g_id and e["target"] == f_id]
        assert len(matching) == 1
        assert matching[0]["line"] == 6


class TestExternalNodes:
    def test_created_on_first_import(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "import requests\n", settings, tokenizer)
        resolve_repo(conn)

        row = conn.execute(
            "SELECT kind, name, qualified_name, file_path, language FROM nodes WHERE id = ?",
            ("external::requests",),
        ).fetchone()
        assert row is not None
        assert row["kind"] == "external_module"
        assert row["name"] == "requests"
        assert row["qualified_name"] == "requests"
        assert row["file_path"] == ""
        assert row["language"] is None

        hit = conn.execute("SELECT id FROM nodes_fts WHERE nodes_fts MATCH 'requests'").fetchall()
        assert any(r["id"] == "external::requests" for r in hit)

    def test_removed_when_no_longer_imported(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "import requests\n", settings, tokenizer)
        resolve_repo(conn)
        assert conn.execute("SELECT 1 FROM nodes WHERE id = 'external::requests'").fetchone() is not None

        sync_source(conn, "a.py", "x = 1\n", settings, tokenizer)
        resolve_repo(conn)

        assert conn.execute("SELECT 1 FROM nodes WHERE id = 'external::requests'").fetchone() is None
        assert conn.execute("SELECT 1 FROM nodes_fts WHERE id = 'external::requests'").fetchone() is None

    def test_one_node_per_package_across_languages(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "import yaml\n", settings, tokenizer)
        sync_source(conn, "b.ts", "import yaml from 'yaml';\n", settings, tokenizer)
        resolve_repo(conn)

        count = conn.execute("SELECT COUNT(*) AS n FROM nodes WHERE id = 'external::yaml'").fetchone()["n"]
        assert count == 1

        py_file_id = file_node_id(conn, "a.py")
        ts_file_id = file_node_id(conn, "b.ts")
        imports = edge_set(conn, kind="imports")
        assert (py_file_id, "external::yaml", "imports", 1, "imported") in imports
        assert (ts_file_id, "external::yaml", "imports", 1, "imported") in imports

    def test_external_nodes_have_no_vectors(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "import requests\n", settings, tokenizer)
        resolve_repo(conn)

        (rowid,) = conn.execute("SELECT rowid FROM nodes WHERE id = 'external::requests'").fetchone()
        assert conn.execute("SELECT 1 FROM vectors WHERE node_rowid = ?", (rowid,)).fetchone() is None


class TestDegree:
    def test_degree_of_a_called_function(self, conn, settings, tokenizer):
        sync_source(
            conn,
            "a.py",
            "def helper():\n    return other()\n\n\ndef other():\n    return 1\n\n\n"
            "def caller1():\n    return helper()\n\n\ndef caller2():\n    return helper()\n",
            settings,
            tokenizer,
        )
        sync_source(conn, "b.py", "def caller3():\n    return helper()\n", settings, tokenizer)
        resolve_repo(conn)

        helper_id = node_id(conn, "a.py", "helper", "function")
        assert degree(conn, helper_id) == 4

    def test_containment_does_not_add_degree(self, conn, settings, tokenizer):
        methods = "\n".join(f"    def m{i}(self):\n        return 0\n" for i in range(10))
        sync_source(conn, "a.py", f"class C:\n{methods}", settings, tokenizer)
        resolve_repo(conn)

        class_id = node_id(conn, "a.py", "C", "class")
        assert degree(conn, class_id) == 0

    def test_degree_follows_edge_changes(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "def helper():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "b.py", "def caller1():\n    return helper()\n", settings, tokenizer)
        sync_source(conn, "c.py", "def caller2():\n    return helper()\n", settings, tokenizer)
        resolve_repo(conn)

        helper_id = node_id(conn, "a.py", "helper", "function")
        assert degree(conn, helper_id) == 2

        sync_source(conn, "c.py", "def caller2():\n    return 1\n", settings, tokenizer)
        resolve_repo(conn)

        assert degree(conn, helper_id) == 1


class TestGraphConsistency:
    def test_no_dangling_edges_after_deletion(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "def helper():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "b.py", "def caller():\n    return helper()\n", settings, tokenizer)
        resolve_repo(conn)
        helper_id = node_id(conn, "a.py", "helper", "function")
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM edges WHERE target = ? AND kind = 'calls'", (helper_id,)
            ).fetchone()["n"]
            == 1
        )

        remove_file(conn, "a.py")
        resolve_repo(conn)

        assert conn.execute("SELECT COUNT(*) AS n FROM edges WHERE target = ?", (helper_id,)).fetchone()["n"] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM refs WHERE resolved_target_id = ?", (helper_id,)
            ).fetchone()["n"]
            == 0
        )
        ref = ref_row(conn, "b.py", "caller")
        assert ref["status"] == "failed"

    def test_unique_becomes_ambiguous_updates_edge_confidence(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "def thing():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "caller.py", "def run():\n    return thing()\n", settings, tokenizer)
        resolve_repo(conn)

        run_id = node_id(conn, "caller.py", "run", "function")
        thing_a_id = node_id(conn, "a.py", "thing", "function")
        assert (run_id, thing_a_id, "calls", 2, "unique_name") in edge_set(conn, kind="calls")
        edge_id_before = next(r["id"] for r in edge_rows(conn, kind="calls") if r["target"] == thing_a_id)

        sync_source(conn, "b.py", "def thing():\n    return 2\n", settings, tokenizer)
        resolve_repo(conn)

        thing_b_id = node_id(conn, "b.py", "thing", "function")
        calls = edge_set(conn, kind="calls")
        assert (run_id, thing_a_id, "calls", 2, "ambiguous") in calls
        assert (run_id, thing_b_id, "calls", 2, "ambiguous") in calls
        edge_id_after = next(r["id"] for r in edge_rows(conn, kind="calls") if r["target"] == thing_a_id)
        assert edge_id_after == edge_id_before

    def test_unchanged_edges_keep_row_ids(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "def helper():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "b.py", "def caller():\n    return helper()\n", settings, tokenizer)
        resolve_repo(conn)
        before = {(r["source"], r["target"], r["kind"], r["line"]): r["id"] for r in edge_rows(conn)}

        sync_source(conn, "c.py", "def unrelated():\n    return 1\n", settings, tokenizer)
        resolve_repo(conn)
        after = {(r["source"], r["target"], r["kind"], r["line"]): r["id"] for r in edge_rows(conn)}

        for key, edge_id in before.items():
            assert after[key] == edge_id

    def test_incremental_matches_from_scratch(self, tmp_path, settings, tokenizer):
        from indexter.paths import canonical_repo_path

        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()
        db_a = tmp_path / "a.db"
        with open_db(db_a, repo=repo_a, settings=settings) as conn_a:
            sync_source(conn_a, "a.py", "def helper():\n    return 1\n", settings, tokenizer)
            resolve_repo(conn_a)
            sync_source(conn_a, "b.py", "def caller():\n    return helper()\n", settings, tokenizer)
            resolve_repo(conn_a)
            sync_source(conn_a, "a.py", "def helper():\n    return 2\n", settings, tokenizer)
            resolve_repo(conn_a)
            incremental_edges = edge_set(conn_a)
            incremental_refs = {
                (r["raw_name"], r["status"], r["resolved_target_id"], r["confidence"])
                for r in conn_a.execute("SELECT raw_name, status, resolved_target_id, confidence FROM refs")
            }

        repo_b = tmp_path / "repo-b"
        repo_b.mkdir()
        db_b = tmp_path / "b.db"
        with open_db(db_b, repo=repo_b, settings=settings) as conn_b:
            sync_source(conn_b, "a.py", "def helper():\n    return 2\n", settings, tokenizer)
            sync_source(conn_b, "b.py", "def caller():\n    return helper()\n", settings, tokenizer)
            resolve_repo(conn_b)
            from_scratch_edges = edge_set(conn_b)
            from_scratch_refs = {
                (r["raw_name"], r["status"], r["resolved_target_id"], r["confidence"])
                for r in conn_b.execute("SELECT raw_name, status, resolved_target_id, confidence FROM refs")
            }

        assert incremental_edges == from_scratch_edges
        assert incremental_refs == from_scratch_refs
        assert canonical_repo_path(repo_a) != canonical_repo_path(repo_b)


class TestResolveRepoFailure:
    def test_failure_mid_resolution_rolls_back(self, conn, settings, tokenizer, monkeypatch):
        sync_source(conn, "a.py", "def helper():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "b.py", "def caller():\n    return helper()\n", settings, tokenizer)

        before_refs = conn.execute("SELECT * FROM refs ORDER BY id").fetchall()
        before_edges = conn.execute("SELECT * FROM edges ORDER BY id").fetchall()

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(graph_mod, "_sync_external_nodes", boom)
        with pytest.raises(RuntimeError, match="boom"):
            resolve_repo(conn)

        after_refs = conn.execute("SELECT * FROM refs ORDER BY id").fetchall()
        after_edges = conn.execute("SELECT * FROM edges ORDER BY id").fetchall()
        assert [dict(r) for r in after_refs] == [dict(r) for r in before_refs]
        assert [dict(r) for r in after_edges] == [dict(r) for r in before_edges]
        pending = conn.execute(
            "SELECT value FROM project_metadata WHERE key = 'resolution_pending'"
        ).fetchone()["value"]
        assert pending == "1"


class TestResolutionSummary:
    def test_summary_matches_stored_rows(self, conn, settings, tokenizer):
        sync_source(conn, "a.py", "import requests\n", settings, tokenizer)
        sync_source(conn, "b.py", "def helper():\n    return 1\n", settings, tokenizer)
        sync_source(conn, "c.py", "def caller():\n    return helper()\n", settings, tokenizer)
        resolve_repo(conn)

        summary = resolution_summary(conn)

        expected_refs = {
            (row["ref_kind"], row["status"], row["confidence"]): row["n"]
            for row in conn.execute(
                "SELECT ref_kind, status, confidence, COUNT(*) AS n FROM refs GROUP BY ref_kind, status, confidence"
            )
        }
        expected_edges = {
            (row["kind"], row["confidence"]): row["n"]
            for row in conn.execute("SELECT kind, confidence, COUNT(*) AS n FROM edges GROUP BY kind, confidence")
        }
        assert summary.refs_by_outcome == expected_refs
        assert summary.edges_by_kind_confidence == expected_edges
        assert summary.external_node_count == 1
