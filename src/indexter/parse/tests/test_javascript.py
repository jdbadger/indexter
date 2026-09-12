from pathlib import Path

import pytest

from indexter.parse.javascript import JavaScriptParser
from indexter.parse.models import Kind, RefKind

FIXTURES = Path(__file__).parent / "fixtures" / "javascript"


@pytest.fixture
def parser():
    return JavaScriptParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.js").read_text()
        result = parser.parse("javascript/sample.js", content)
        kinds = {n.name: n.kind for n in result.nodes if n.name}
        assert kinds["Base"] == Kind.CLASS
        assert kinds["Handler"] == Kind.CLASS
        assert kinds["process"] == Kind.METHOD
        assert kinds["standalone"] == Kind.FUNCTION
        assert kinds["double"] == Kind.FUNCTION
        assert kinds["onItem"] == Kind.FUNCTION  # nested in a method, not itself a class member

    def test_no_import_or_export_nodes(self, parser):
        content = (FIXTURES / "sample.js").read_text()
        result = parser.parse("javascript/sample.js", content)
        names = {n.name for n in result.nodes}
        assert "events" not in names
        assert "defaultExport" not in names

    def test_byte_ranges_slice_back_to_exact_source(self, parser):
        content = (FIXTURES / "sample.js").read_text()
        result = parser.parse("javascript/sample.js", content)
        source_bytes = content.encode()
        node = next(n for n in result.nodes if n.name == "standalone")
        text = source_bytes[node.start_byte : node.end_byte].decode()
        assert text.startswith("function standalone(x)")

    def test_empty_file(self, parser):
        result = parser.parse("a.js", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]
        assert result.errors == []

    def test_uppercase_const_is_a_constant(self, parser):
        result = parser.parse("a.js", "const MAX_RETRIES = 3;\n")
        node = next(n for n in result.nodes if n.name == "MAX_RETRIES")
        assert node.kind == Kind.CONSTANT

    def test_lowercase_const_non_function_is_not_a_node(self, parser):
        result = parser.parse("a.js", "const total = 3;\n")
        assert all(n.name != "total" for n in result.nodes)


class TestCollisions:
    def test_named_callback_scoped_to_class_and_method(self, parser):
        content = (FIXTURES / "sample.js").read_text()
        result = parser.parse("javascript/sample.js", content)
        callback = next(n for n in result.nodes if n.name == "onItem")
        assert callback.scope_path == ("Handler", "process")

    def test_two_object_literal_handlers_are_distinct(self, parser):
        content = (FIXTURES / "objects.js").read_text()
        result = parser.parse("javascript/objects.js", content)
        handlers = [n for n in result.nodes if n.name == "handler"]
        assert len(handlers) == 2
        assert {h.scope_path for h in handlers} == {("first",), ("second",)}
        assert handlers[0].id != handlers[1].id
        assert "~" not in handlers[0].id
        assert "~" not in handlers[1].id

    def test_arrow_function_assigned_to_const_has_no_self_referential_scope(self, parser):
        result = parser.parse("a.js", "const double = (x) => x * 2;\n")
        node = next(n for n in result.nodes if n.name == "double")
        assert node.scope_path == ()


class TestSignaturesAndDocs:
    def test_unparenthesized_single_param_arrow_signature(self, parser):
        result = parser.parse("a.js", "const f = x => x * 2;\n")
        node = next(n for n in result.nodes if n.name == "f")
        assert node.signature == "x =>"

    def test_generator_function(self, parser):
        result = parser.parse("a.js", "function* gen() { yield 1; }\n")
        node = next(n for n in result.nodes if n.name == "gen")
        assert node.kind == Kind.FUNCTION

    def test_jsdoc_comment_is_parsed(self, parser):
        content = "/**\n * Does the thing.\n * @param x input\n */\nfunction f(x) {}\n"
        result = parser.parse("a.js", content)
        node = next(n for n in result.nodes if n.name == "f")
        assert node.docstring == "Does the thing.\n@param x input"

    def test_non_jsdoc_comment_is_ignored(self, parser):
        content = "// just a regular comment\nfunction f() {}\n"
        result = parser.parse("a.js", content)
        node = next(n for n in result.nodes if n.name == "f")
        assert node.docstring is None

    def test_method_definition_signature(self, parser):
        result = parser.parse("a.js", "class C { m(x) { return x; } }")
        node = next(n for n in result.nodes if n.name == "m")
        assert node.signature == "m(x)"


class TestDefensiveBranches:
    """Unit-tests match-handler fallbacks that real tree-sitter queries
    never actually produce (every capture group they emit is handled), so
    these are only reachable via direct calls.
    """

    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_process_definition_match_without_name_capture(self, parser):
        # A def capture with no name -- can't happen via our query since
        # every definitions pattern requires a name field, but the guard
        # exists for safety.
        class FakeNode:
            type = "function_declaration"

        assert parser.process_definition_match({"def": [FakeNode()]}, b"") is None

    def test_process_reference_match_with_no_recognized_capture(self, parser):
        assert parser.process_reference_match({}, b"") is None

    def test_jsdoc_with_no_parent(self, parser):
        class FakeNode:
            parent = None

        assert parser._jsdoc(FakeNode(), b"") is None

    def test_arrow_signature_with_no_params_field_falls_back_to_body_start(self, parser):
        # Every real arrow_function has a `parameters`/`parameter` field,
        # but the fallback exists for safety -- exercised directly.
        class FakeBody:
            start_byte = 5

        class FakeNode:
            type = "arrow_function"
            start_byte = 0
            children = []

            def child_by_field_name(self, name):
                return FakeBody() if name == "body" else None

        assert parser._signature(FakeNode(), b"12345body") == "12345"

    def test_arrow_signature_with_neither_params_nor_body_is_none(self, parser):
        class FakeNode:
            type = "arrow_function"
            start_byte = 0
            children = []

            def child_by_field_name(self, name):
                return None

        assert parser._signature(FakeNode(), b"") is None


class TestReferences:
    def test_member_chain_call_head(self, parser):
        result = parser.parse("a.js", "os.path.join(a, b);\n")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.CALLS]
        assert ref.raw_name == "os.path.join"
        assert ref.head == "os"

    def test_this_chain_call_head(self, parser):
        result = parser.parse("a.js", "class C { m() { this.x.y(); } }")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.CALLS]
        assert ref.raw_name == "this.x.y"
        assert ref.head == "this"

    def test_chained_call_has_no_head(self, parser):
        # `build().run()` is two calls: the outer `.run()` (no head, since
        # its base is itself a call) and the inner `build()` (head "build").
        result = parser.parse("a.js", "build().run();\n")
        calls = {r.raw_name: r.head for r in result.refs if r.ref_kind == RefKind.CALLS}
        assert calls == {"build().run": None, "build": "build"}

    def test_bare_call_head_equals_raw_name(self, parser):
        result = parser.parse("a.js", "helper();\n")
        [ref] = result.refs
        assert ref.raw_name == "helper"
        assert ref.head == "helper"

    def test_es_import_specifier(self, parser):
        result = parser.parse("a.js", 'import { X } from "./mod";\n')
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        assert ref.raw_name == "./mod"

    def test_require_specifier(self, parser):
        result = parser.parse("a.js", 'const x = require("./mod");\n')
        imports = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        assert any(r.raw_name == "./mod" for r in imports)

    def test_extends_yields_inherits_ref(self, parser):
        content = (FIXTURES / "sample.js").read_text()
        result = parser.parse("javascript/sample.js", content)
        inherits = [r for r in result.refs if r.ref_kind == RefKind.INHERITS]
        assert {r.raw_name for r in inherits} == {"EventEmitter", "Base"}

    def test_every_ref_origin_matches_a_node_from_the_same_parse(self, parser):
        content = (FIXTURES / "sample.js").read_text()
        result = parser.parse("javascript/sample.js", content)
        node_ids = {n.id for n in result.nodes}
        assert result.refs
        assert all(r.from_node_id in node_ids for r in result.refs)
