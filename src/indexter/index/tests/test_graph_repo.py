"""End-to-end tests over the multi-language fixture at
`fixtures/graph_repo/` (design.md decision 13): syncing it once with
`FakeEmbedder` and asserting the specific edges each resolution tier and
language should produce (tasks.md 7.2), plus an inline snapshot of every
edge and every reference outcome so a regression in any tier shows up as a
diff instead of passing unnoticed (tasks.md 7.3).

Where `test_graph.py` builds small in-memory scenarios for each rule in
isolation, this exercises the whole pipeline -- real parsing, real module
resolution, real graph derivation -- over one repository that combines
Python, JavaScript/TypeScript and Rust.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from indexter.config import Settings
from indexter.db.connection import open_db
from indexter.index.embed import FakeEmbedder
from indexter.index.sync import sync_repo

FIXTURE = Path(__file__).parent / "fixtures" / "graph_repo"


@pytest.fixture(scope="module")
def synced(tmp_path_factory):
    settings = Settings(embedding_dim=4, embed_max_tokens=256)
    embedder = FakeEmbedder(dim=4)
    db_path = tmp_path_factory.mktemp("graph_repo_db") / "test.db"
    with open_db(db_path, repo=FIXTURE, settings=settings) as conn:
        report = sync_repo(conn, FIXTURE, settings, embedder)
        yield conn, report


@pytest.fixture
def conn(synced):
    return synced[0]


def node_id(conn, path: str, name: str, kind: str | None = None) -> str:
    query = "SELECT id FROM nodes WHERE file_path = ? AND name = ?"
    params: list[str] = [path, name]
    if kind is not None:
        query += " AND kind = ?"
        params.append(kind)
    row = conn.execute(query, params).fetchone()
    assert row is not None, f"no {kind or 'node'} named {name!r} in {path!r}"
    return row["id"]


def edge_set(conn, **where):
    query = "SELECT source, target, kind, line, confidence FROM edges"
    if where:
        query += " WHERE " + " AND ".join(f"{k} = ?" for k in where)
    rows = conn.execute(query, list(where.values())).fetchall()
    return {(r["source"], r["target"], r["kind"], r["line"], r["confidence"]) for r in rows}


def ref_status(conn, from_path: str, raw_name: str) -> tuple[str, str | None]:
    row = conn.execute(
        "SELECT refs.status, refs.confidence FROM refs JOIN nodes ON nodes.id = refs.from_node_id "
        "WHERE nodes.file_path = ? AND refs.raw_name = ?",
        (from_path, raw_name),
    ).fetchone()
    assert row is not None, f"no ref {raw_name!r} from {from_path!r}"
    return row["status"], row["confidence"]


class TestReportedNoErrors:
    def test_every_fixture_file_parses_without_error(self, synced):
        _conn, report = synced
        assert report.errors == {}


class TestPythonEdges:
    def test_tier1_inherited_self_call(self, conn):
        run_id = node_id(conn, "python/pkg/core.py", "run", "method")
        save_id = node_id(conn, "python/pkg/core.py", "save", "method")
        assert (run_id, save_id, "calls", 11, "exact") in edge_set(conn, kind="calls")

    def test_tier2_nested_function(self, conn):
        outer_id = node_id(conn, "python/pkg/core.py", "make_counter", "function")
        inner_id = node_id(conn, "python/pkg/core.py", "increment", "function")
        assert (outer_id, inner_id, "calls", 22, "exact") in edge_set(conn, kind="calls")

    def test_tier2_shadowed_builtin_resolves_locally(self, conn):
        use_len_id = node_id(conn, "python/app.py", "use_len", "function")
        local_len_id = node_id(conn, "python/app.py", "len", "function")
        assert (use_len_id, local_len_id, "calls", 15, "exact") in edge_set(conn, kind="calls")

    def test_tier3_package_re_export(self, conn):
        build_id = node_id(conn, "python/app.py", "build", "function")
        engine_id = node_id(conn, "python/pkg/core.py", "Engine", "class")
        assert (build_id, engine_id, "calls", 23, "imported") in edge_set(conn, kind="calls")

    def test_tier3_aliased_import(self, conn):
        build_id = node_id(conn, "python/app.py", "build", "function")
        helper_id = node_id(conn, "python/pkg/util.py", "helper", "function")
        assert (build_id, helper_id, "calls", 24, "imported") in edge_set(conn, kind="calls")

    def test_tier3_wildcard_import(self, conn):
        build_id = node_id(conn, "python/app.py", "build", "function")
        core_parse_id = node_id(conn, "python/pkg/core.py", "parse", "function")
        assert (build_id, core_parse_id, "calls", 24, "imported") in edge_set(conn, kind="calls")

    def test_tier3_relative_import_across_subpackage(self, conn):
        use_helper_id = node_id(conn, "python/pkg/sub/mod_a.py", "use_helper", "function")
        helper_id = node_id(conn, "python/pkg/util.py", "helper", "function")
        assert (use_helper_id, helper_id, "calls", 7, "imported") in edge_set(conn, kind="calls")

    def test_tier5_ambiguous_across_two_modules(self, conn):
        caller_id = node_id(conn, "python/app.py", "ambiguous_call", "function")
        core_parse_id = node_id(conn, "python/pkg/core.py", "parse", "function")
        util_parse_id = node_id(conn, "python/pkg/util.py", "parse", "function")
        assert edge_set(conn, source=caller_id) == {
            (caller_id, core_parse_id, "calls", 19, "ambiguous"),
            (caller_id, util_parse_id, "calls", 19, "ambiguous"),
        }

    def test_inheritance_edge(self, conn):
        engine_id = node_id(conn, "python/pkg/core.py", "Engine", "class")
        base_id = node_id(conn, "python/pkg/core.py", "Base", "class")
        assert (engine_id, base_id, "inherits", 9, "exact") in edge_set(conn, kind="inherits")


class TestJavaScriptTypeScriptEdges:
    def test_default_import_with_extension_rewrite(self, conn):
        use_all_id = node_id(conn, "js/app.js", "useAll", "function")
        multiply_id = node_id(conn, "js/utils/math.ts", "multiply", "function")
        assert (use_all_id, multiply_id, "calls", 7, "imported") in edge_set(conn, kind="calls")

    def test_named_import_through_barrel_index(self, conn):
        use_all_id = node_id(conn, "js/app.js", "useAll", "function")
        add_id = node_id(conn, "js/utils/math.ts", "add", "function")
        assert (use_all_id, add_id, "calls", 7, "imported") in edge_set(conn, kind="calls")

    def test_namespace_import_member_access(self, conn):
        use_all_id = node_id(conn, "js/app.js", "useAll", "function")
        add_id = node_id(conn, "js/utils/math.ts", "add", "function")
        assert (use_all_id, add_id, "calls", 7, "imported") in edge_set(conn, kind="calls")

    def test_callback_body_resolves_enclosing_scope(self, conn):
        process_all_id = node_id(conn, "js/callbacks.js", "processAll", "function")
        transform_id = node_id(conn, "js/callbacks.js", "transform", "function")
        assert (process_all_id, transform_id, "calls", 3, "exact") in edge_set(conn, kind="calls")

    def test_this_call_resolves_to_own_method(self, conn):
        greet_id = node_id(conn, "js/components/handler.ts", "greet", "method")
        format_id = node_id(conn, "js/components/handler.ts", "format", "method")
        assert (greet_id, format_id, "calls", 6, "exact") in edge_set(conn, kind="calls")

    def test_extends_and_implements(self, conn):
        handler_id = node_id(conn, "js/components/handler.ts", "Handler", "class")
        base_id = node_id(conn, "js/components/base.ts", "Base", "class")
        greeter_id = node_id(conn, "js/components/base.ts", "Greeter", "interface")
        assert edge_set(conn, source=handler_id, kind="inherits") == {
            (handler_id, base_id, "inherits", 4, "imported"),
            (handler_id, greeter_id, "inherits", 4, "imported"),
        }

    def test_barrel_re_export_targets_the_original_definitions(self, conn):
        index_id = node_id(conn, "js/utils/index.ts", "", "file")
        add_id = node_id(conn, "js/utils/math.ts", "add", "function")
        multiply_id = node_id(conn, "js/utils/math.ts", "multiply", "function")
        assert edge_set(conn, source=index_id, kind="imports") == {
            (index_id, add_id, "imports", 1, "imported"),
            (index_id, multiply_id, "imports", 2, "imported"),
        }


class TestRustEdges:
    def test_crate_rooted_use(self, conn):
        main_file_id = node_id(conn, "rust/src/main.rs", "", "file")
        handler_id = node_id(conn, "rust/src/auth/mod.rs", "Handler", "struct")
        assert (main_file_id, handler_id, "imports", 6, "imported") in edge_set(conn, kind="imports")

    def test_super_path_call(self, conn):
        client_login_id = node_id(conn, "rust/src/net/client.rs", "client_login", "function")
        retry_id = node_id(conn, "rust/src/net/mod.rs", "retry", "function")
        assert (client_login_id, retry_id, "calls", 6, "imported") in edge_set(conn, kind="calls")

    def test_self_path_call(self, conn):
        ping_id = node_id(conn, "rust/src/net/mod.rs", "ping", "function")
        retry_id = node_id(conn, "rust/src/net/mod.rs", "retry", "function")
        assert (ping_id, retry_id, "calls", 8, "imported") in edge_set(conn, kind="calls")

    def test_self_new_associated_function(self, conn):
        reset_id = node_id(conn, "rust/src/auth/mod.rs", "reset", "method")
        new_id = node_id(conn, "rust/src/auth/mod.rs", "new", "method")
        assert (reset_id, new_id, "calls", 11, "exact") in edge_set(conn, kind="calls")

    def test_unique_method_name_across_the_crate(self, conn):
        login_id = node_id(conn, "rust/src/auth/mod.rs", "login", "method")
        for caller_path, caller_name, line in (
            ("rust/src/main.rs", "main", 10),
            ("rust/src/net/client.rs", "client_login", 5),
        ):
            caller_id = node_id(conn, caller_path, caller_name, "function")
            assert (caller_id, login_id, "calls", line, "unique_name") in edge_set(conn, kind="calls")

    def test_trait_impl_for_type_in_another_file(self, conn):
        """`impl Greet for Foo` lives in `fmt_impls.rs`, `Foo` in `model.rs`
        (design.md decision 4): the edge's source is `Foo`, not the file the
        `impl` block is written in."""
        foo_id = node_id(conn, "rust/src/model.rs", "Foo", "struct")
        greet_id = node_id(conn, "rust/src/fmt_impls.rs", "Greet", "trait")
        assert (foo_id, greet_id, "inherits", 9, "exact") in edge_set(conn, kind="inherits")

    def test_external_trait_impl_still_attributes_for_type_but_yields_no_edge(self, conn):
        """`impl Debug for Foo` resolves `Debug` to `external::std` (decision
        7: an `external` reference produces an edge only when its kind is
        `imports`) -- the ref is `external`, not `failed`, but no `inherits`
        edge exists for it."""
        status, confidence = ref_status(conn, "rust/src/fmt_impls.rs", "Debug")
        assert (status, confidence) == ("external", "imported")

        foo_id = node_id(conn, "rust/src/model.rs", "Foo", "struct")
        assert all(edge[2] != "inherits" or edge[1] != "external::std" for edge in edge_set(conn, source=foo_id))


class TestExternalModules:
    def test_std_import_creates_one_external_node(self, conn):
        row = conn.execute(
            "SELECT kind, name, qualified_name, file_path, language FROM nodes WHERE id = 'external::std'"
        ).fetchone()
        assert dict(row) == {
            "kind": "external_module",
            "name": "std",
            "qualified_name": "std",
            "file_path": "",
            "language": None,
        }

    def test_external_module_has_no_vector(self, conn):
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM vectors v JOIN nodes n ON n.rowid = v.node_rowid WHERE n.id = 'external::std'"
        ).fetchone()
        assert count == 0


def _render_graph(conn) -> str:
    lines = ["EDGES:"]
    for row in conn.execute(
        "SELECT source, target, kind, line, confidence FROM edges ORDER BY kind, source, target, IFNULL(line, -1)"
    ):
        lines.append(f"{row['kind']} {row['source']} -> {row['target']} line={row['line']} conf={row['confidence']}")

    lines.append("")
    lines.append("REF OUTCOMES:")
    for row in conn.execute(
        "SELECT n.file_path AS from_file, r.raw_name, r.ref_kind, r.line, r.status, "
        "r.resolved_target_id, r.confidence, r.candidates "
        "FROM refs r JOIN nodes n ON n.id = r.from_node_id "
        "ORDER BY n.file_path, r.line, r.raw_name"
    ):
        candidates = f" candidates={row['candidates']}" if row["candidates"] else ""
        lines.append(
            f"{row['from_file']}:{row['line']} {row['ref_kind']} {row['raw_name']!r} -> "
            f"{row['status']}"
            f"{' ' + row['resolved_target_id'] if row['resolved_target_id'] else ''}"
            f"{' (' + row['confidence'] + ')' if row['confidence'] else ''}"
            f"{candidates}"
        )
    return "\n".join(lines)


class TestFullGraphSnapshot:
    def test_every_edge_and_ref_outcome(self, conn):
        assert _render_graph(conn) == snapshot("""\
EDGES:
calls js/app.js::useAll#function -> js/utils/math.ts::add#function line=7 conf=imported
calls js/app.js::useAll#function -> js/utils/math.ts::multiply#function line=7 conf=imported
calls js/callbacks.js::processAll#function -> js/callbacks.js::transform#function line=3 conf=exact
calls js/components/handler.ts::Handler.greet#method -> js/components/handler.ts::Handler.format#method line=6 conf=exact
calls python/app.py::ambiguous_call#function -> python/pkg/core.py::parse#function line=19 conf=ambiguous
calls python/app.py::ambiguous_call#function -> python/pkg/util.py::parse#function line=19 conf=ambiguous
calls python/app.py::build#function -> python/app.py::use_len#function line=24 conf=exact
calls python/app.py::build#function -> python/pkg/core.py::Engine#class line=23 conf=imported
calls python/app.py::build#function -> python/pkg/core.py::parse#function line=24 conf=imported
calls python/app.py::build#function -> python/pkg/util.py::helper#function line=24 conf=imported
calls python/app.py::use_len#function -> python/app.py::len#function line=15 conf=exact
calls python/pkg/core.py::Engine.run#method -> python/pkg/core.py::Base.save#method line=11 conf=exact
calls python/pkg/core.py::make_counter#function -> python/pkg/core.py::make_counter.increment#function line=22 conf=exact
calls python/pkg/sub/mod_a.py::use_helper#function -> python/pkg/util.py::helper#function line=7 conf=imported
calls rust/src/auth/mod.rs::Handler.login#method -> rust/src/auth/mod.rs::Handler.validate#method line=15 conf=exact
calls rust/src/auth/mod.rs::Handler.reset#method -> rust/src/auth/mod.rs::Handler.new#method line=11 conf=exact
calls rust/src/main.rs::main#function -> rust/src/auth/mod.rs::Handler.login#method line=10 conf=unique_name
calls rust/src/main.rs::main#function -> rust/src/auth/mod.rs::Handler.new#method line=9 conf=imported
calls rust/src/net/client.rs::client_login#function -> rust/src/auth/mod.rs::Handler.login#method line=5 conf=unique_name
calls rust/src/net/client.rs::client_login#function -> rust/src/auth/mod.rs::Handler.new#method line=4 conf=imported
calls rust/src/net/client.rs::client_login#function -> rust/src/net/mod.rs::retry#function line=6 conf=imported
calls rust/src/net/mod.rs::ping#function -> rust/src/net/mod.rs::retry#function line=8 conf=imported
contains js/app.js::#file -> js/app.js::useAll#function line=None conf=exact
contains js/callbacks.js::#file -> js/callbacks.js::processAll#function line=None conf=exact
contains js/callbacks.js::#file -> js/callbacks.js::transform#function line=None conf=exact
contains js/components/base.ts::#file -> js/components/base.ts::Base#class line=None conf=exact
contains js/components/base.ts::#file -> js/components/base.ts::Greeter#interface line=None conf=exact
contains js/components/base.ts::Base#class -> js/components/base.ts::Base.setup#method line=None conf=exact
contains js/components/base.ts::Greeter#interface -> js/components/base.ts::Greeter.greet#method line=None conf=exact
contains js/components/handler.ts::#file -> js/components/handler.ts::Handler#class line=None conf=exact
contains js/components/handler.ts::Handler#class -> js/components/handler.ts::Handler.format#method line=None conf=exact
contains js/components/handler.ts::Handler#class -> js/components/handler.ts::Handler.greet#method line=None conf=exact
contains js/utils/math.ts::#file -> js/utils/math.ts::add#function line=None conf=exact
contains js/utils/math.ts::#file -> js/utils/math.ts::multiply#function line=None conf=exact
contains python/app.py::#file -> python/app.py::ambiguous_call#function line=None conf=exact
contains python/app.py::#file -> python/app.py::build#function line=None conf=exact
contains python/app.py::#file -> python/app.py::len#function line=None conf=exact
contains python/app.py::#file -> python/app.py::use_len#function line=None conf=exact
contains python/pkg/core.py::#file -> python/pkg/core.py::Base#class line=None conf=exact
contains python/pkg/core.py::#file -> python/pkg/core.py::Engine#class line=None conf=exact
contains python/pkg/core.py::#file -> python/pkg/core.py::make_counter#function line=None conf=exact
contains python/pkg/core.py::#file -> python/pkg/core.py::parse#function line=None conf=exact
contains python/pkg/core.py::Base#class -> python/pkg/core.py::Base.save#method line=None conf=exact
contains python/pkg/core.py::Engine#class -> python/pkg/core.py::Engine.run#method line=None conf=exact
contains python/pkg/core.py::make_counter#function -> python/pkg/core.py::make_counter.increment#function line=None conf=exact
contains python/pkg/sub/mod_a.py::#file -> python/pkg/sub/mod_a.py::use_helper#function line=None conf=exact
contains python/pkg/util.py::#file -> python/pkg/util.py::helper#function line=None conf=exact
contains python/pkg/util.py::#file -> python/pkg/util.py::parse#function line=None conf=exact
contains rust/src/auth/mod.rs::#file -> rust/src/auth/mod.rs::Handler#struct line=None conf=exact
contains rust/src/auth/mod.rs::#file -> rust/src/auth/mod.rs::Handler.login#method line=None conf=exact
contains rust/src/auth/mod.rs::#file -> rust/src/auth/mod.rs::Handler.new#method line=None conf=exact
contains rust/src/auth/mod.rs::#file -> rust/src/auth/mod.rs::Handler.reset#method line=None conf=exact
contains rust/src/auth/mod.rs::#file -> rust/src/auth/mod.rs::Handler.validate#method line=None conf=exact
contains rust/src/fmt_impls.rs::#file -> rust/src/fmt_impls.rs::Foo<Debug>.fmt#method line=None conf=exact
contains rust/src/fmt_impls.rs::#file -> rust/src/fmt_impls.rs::Foo<Greet>.greet#method line=None conf=exact
contains rust/src/fmt_impls.rs::#file -> rust/src/fmt_impls.rs::Greet#trait line=None conf=exact
contains rust/src/main.rs::#file -> rust/src/main.rs::main#function line=None conf=exact
contains rust/src/model.rs::#file -> rust/src/model.rs::Foo#struct line=None conf=exact
contains rust/src/net/client.rs::#file -> rust/src/net/client.rs::client_login#function line=None conf=exact
contains rust/src/net/mod.rs::#file -> rust/src/net/mod.rs::ping#function line=None conf=exact
contains rust/src/net/mod.rs::#file -> rust/src/net/mod.rs::retry#function line=None conf=exact
imports js/app.js::#file -> js/components/handler.ts::#file line=4 conf=imported
imports js/app.js::#file -> js/utils/math.ts::#file line=3 conf=imported
imports js/app.js::#file -> js/utils/math.ts::add#function line=2 conf=imported
imports js/app.js::#file -> js/utils/math.ts::multiply#function line=1 conf=imported
imports js/components/handler.ts::#file -> js/components/base.ts::Base#class line=1 conf=imported
imports js/components/handler.ts::#file -> js/components/base.ts::Greeter#interface line=2 conf=imported
imports js/utils/index.ts::#file -> js/utils/math.ts::add#function line=1 conf=imported
imports js/utils/index.ts::#file -> js/utils/math.ts::multiply#function line=2 conf=imported
imports python/app.py::#file -> python/pkg/core.py::#file line=7 conf=imported
imports python/app.py::#file -> python/pkg/core.py::Engine#class line=5 conf=imported
imports python/app.py::#file -> python/pkg/util.py::helper#function line=6 conf=imported
imports python/pkg/__init__.py::#file -> python/pkg/core.py::Engine#class line=3 conf=imported
imports python/pkg/sub/mod_a.py::#file -> python/pkg/util.py::helper#function line=3 conf=imported
imports rust/src/fmt_impls.rs::#file -> external::std line=1 conf=imported
imports rust/src/fmt_impls.rs::#file -> rust/src/model.rs::Foo#struct line=3 conf=imported
imports rust/src/main.rs::#file -> rust/src/auth/mod.rs::Handler#struct line=6 conf=imported
imports rust/src/net/client.rs::#file -> rust/src/auth/mod.rs::Handler#struct line=1 conf=imported
inherits js/components/handler.ts::Handler#class -> js/components/base.ts::Base#class line=4 conf=imported
inherits js/components/handler.ts::Handler#class -> js/components/base.ts::Greeter#interface line=4 conf=imported
inherits python/pkg/core.py::Engine#class -> python/pkg/core.py::Base#class line=9 conf=exact
inherits rust/src/model.rs::Foo#struct -> rust/src/fmt_impls.rs::Greet#trait line=9 conf=exact

REF OUTCOMES:
js/app.js:1 imports './utils/math.js' -> resolved js/utils/math.ts::multiply#function (imported)
js/app.js:2 imports './utils' -> resolved js/utils/math.ts::add#function (imported)
js/app.js:3 imports './utils/math' -> resolved js/utils/math.ts::#file (imported)
js/app.js:4 imports './components/handler' -> resolved js/components/handler.ts::#file (imported)
js/app.js:4 calls 'require' -> failed
js/app.js:7 calls 'add' -> resolved js/utils/math.ts::add#function (imported)
js/app.js:7 calls 'mathNs.add' -> resolved js/utils/math.ts::add#function (imported)
js/app.js:7 calls 'multiply' -> resolved js/utils/math.ts::multiply#function (imported)
js/callbacks.js:2 calls 'items.map' -> failed
js/callbacks.js:3 calls 'transform' -> resolved js/callbacks.js::transform#function (exact)
js/callbacks.js:8 calls 'callback' -> failed
js/components/handler.ts:1 imports './base.js' -> resolved js/components/base.ts::Base#class (imported)
js/components/handler.ts:2 imports './base' -> resolved js/components/base.ts::Greeter#interface (imported)
js/components/handler.ts:4 inherits 'Base' -> resolved js/components/base.ts::Base#class (imported)
js/components/handler.ts:4 inherits 'Greeter' -> resolved js/components/base.ts::Greeter#interface (imported)
js/components/handler.ts:6 calls 'this.format' -> resolved js/components/handler.ts::Handler.format#method (exact)
js/components/handler.ts:10 calls 'name.toUpperCase' -> failed
js/utils/index.ts:1 imports './math' -> resolved js/utils/math.ts::add#function (imported)
js/utils/index.ts:2 imports './math' -> resolved js/utils/math.ts::multiply#function (imported)
python/app.py:5 imports 'pkg' -> resolved python/pkg/core.py::Engine#class (imported)
python/app.py:6 imports 'pkg.util' -> resolved python/pkg/util.py::helper#function (imported)
python/app.py:7 imports 'pkg.core' -> resolved python/pkg/core.py::#file (imported)
python/app.py:15 calls 'len' -> resolved python/app.py::len#function (exact)
python/app.py:19 calls 'x.parse' -> ambiguous (ambiguous) candidates=["python/pkg/core.py::parse#function", "python/pkg/util.py::parse#function"]
python/app.py:23 calls 'Engine' -> resolved python/pkg/core.py::Engine#class (imported)
python/app.py:24 calls 'h' -> resolved python/pkg/util.py::helper#function (imported)
python/app.py:24 calls 'parse' -> resolved python/pkg/core.py::parse#function (imported)
python/app.py:24 calls 'use_len' -> resolved python/app.py::use_len#function (exact)
python/pkg/__init__.py:3 imports '.core' -> resolved python/pkg/core.py::Engine#class (imported)
python/pkg/core.py:9 inherits 'Base' -> resolved python/pkg/core.py::Base#class (exact)
python/pkg/core.py:11 calls 'self.save' -> resolved python/pkg/core.py::Base.save#method (exact)
python/pkg/core.py:22 calls 'increment' -> resolved python/pkg/core.py::make_counter.increment#function (exact)
python/pkg/sub/mod_a.py:3 imports '..util' -> resolved python/pkg/util.py::helper#function (imported)
python/pkg/sub/mod_a.py:7 calls 'helper' -> resolved python/pkg/util.py::helper#function (imported)
rust/src/auth/mod.rs:11 calls 'Self::new' -> resolved rust/src/auth/mod.rs::Handler.new#method (exact)
rust/src/auth/mod.rs:15 calls 'self.validate' -> resolved rust/src/auth/mod.rs::Handler.validate#method (exact)
rust/src/fmt_impls.rs:1 imports 'std::fmt' -> external external::std (imported)
rust/src/fmt_impls.rs:3 imports 'crate::model' -> resolved rust/src/model.rs::Foo#struct (imported)
rust/src/fmt_impls.rs:9 inherits 'Greet' -> resolved rust/src/fmt_impls.rs::Greet#trait (exact)
rust/src/fmt_impls.rs:15 inherits 'Debug' -> external external::std (imported)
rust/src/main.rs:6 imports 'crate::auth' -> resolved rust/src/auth/mod.rs::Handler#struct (imported)
rust/src/main.rs:9 calls 'Handler::new' -> resolved rust/src/auth/mod.rs::Handler.new#method (imported)
rust/src/main.rs:10 calls 'h.login' -> resolved rust/src/auth/mod.rs::Handler.login#method (unique_name)
rust/src/net/client.rs:1 imports 'crate::auth' -> resolved rust/src/auth/mod.rs::Handler#struct (imported)
rust/src/net/client.rs:4 calls 'Handler::new' -> resolved rust/src/auth/mod.rs::Handler.new#method (imported)
rust/src/net/client.rs:5 calls 'h.login' -> resolved rust/src/auth/mod.rs::Handler.login#method (unique_name)
rust/src/net/client.rs:6 calls 'super::retry' -> resolved rust/src/net/mod.rs::retry#function (imported)
rust/src/net/mod.rs:8 calls 'self::retry' -> resolved rust/src/net/mod.rs::retry#function (imported)\
""")
