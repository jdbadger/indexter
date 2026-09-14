from pathlib import Path

import pytest

from indexter.parse.models import Kind, RefKind
from indexter.parse.typescript import TypeScriptParser

FIXTURES = Path(__file__).parent / "fixtures" / "typescript"


@pytest.fixture
def parser():
    return TypeScriptParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        kinds = {n.name: n.kind for n in result.nodes if n.name}
        assert kinds["Greeter"] == Kind.INTERFACE
        assert kinds["Level"] == Kind.TYPE_ALIAS
        assert kinds["Status"] == Kind.ENUM
        assert kinds["BaseHandler"] == Kind.CLASS
        assert kinds["Handler"] == Kind.CLASS
        assert kinds["greet"] == Kind.METHOD  # both Greeter.greet and Handler.greet
        assert kinds["standalone"] == Kind.FUNCTION

    def test_exported_class_yields_exactly_one_node(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        matches = [n for n in result.nodes if n.name == "Handler"]
        assert len(matches) == 1

    def test_exported_function_yields_exactly_one_node(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        matches = [n for n in result.nodes if n.name == "standalone"]
        assert len(matches) == 1

    def test_no_import_or_export_nodes(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        names = {n.name for n in result.nodes}
        assert "export" not in names
        assert "default" not in names

    def test_byte_ranges_slice_back_to_exact_source(self, parser):
        # Unlike Python's decorator (an explicit spec requirement), the
        # node's range is the declaration itself -- "export " is not part
        # of the sliced text, since we capture function_declaration
        # directly rather than the wrapping export_statement.
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        source_bytes = content.encode()
        node = next(n for n in result.nodes if n.name == "standalone")
        text = source_bytes[node.start_byte : node.end_byte].decode()
        assert text.startswith("function standalone")
        assert "return x * 2;" in text

    def test_interface_method_scoped_to_interface(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        greet_methods = [n for n in result.nodes if n.name == "greet"]
        scopes = {n.scope_path for n in greet_methods}
        assert ("Greeter",) in scopes
        assert ("Handler",) in scopes

    def test_empty_file(self, parser):
        result = parser.parse("a.ts", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]
        assert result.errors == []

    def test_default_export_class(self, parser):
        result = parser.parse("a.ts", "export default class Foo {}\n")
        matches = [n for n in result.nodes if n.name == "Foo"]
        assert len(matches) == 1
        assert matches[0].kind == Kind.CLASS


class TestConstantsAndSignatures:
    def test_uppercase_const_is_a_constant(self, parser):
        result = parser.parse("a.ts", "const MAX_RETRIES = 3;\n")
        node = next(n for n in result.nodes if n.name == "MAX_RETRIES")
        assert node.kind == Kind.CONSTANT

    def test_lowercase_const_non_function_is_not_a_node(self, parser):
        result = parser.parse("a.ts", "const total = 3;\n")
        assert all(n.name != "total" for n in result.nodes)

    def test_arrow_function_assigned_to_const(self, parser):
        result = parser.parse("a.ts", "const double = (x: number) => x * 2;\n")
        node = next(n for n in result.nodes if n.name == "double")
        assert node.kind == Kind.FUNCTION
        assert node.signature == "(x: number) =>"

    def test_object_literal_variable_scope_segment(self, parser):
        result = parser.parse("a.ts", "const obj = { handler(y) { return y; } };\n")
        node = next(n for n in result.nodes if n.name == "handler")
        assert node.scope_path == ("obj",)

    def test_require_specifier(self, parser):
        result = parser.parse("a.ts", 'const x = require("./mod");\n')
        imports = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        assert any(r.raw_name == "./mod" for r in imports)

    def test_require_without_assignment_has_no_head(self, parser):
        result = parser.parse("a.ts", 'require("./sideeffect");\n')
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        assert ref.raw_name == "./sideeffect"
        assert ref.head is None

    def test_interface_method_signature_has_no_body(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        greet = next(n for n in result.nodes if n.name == "greet" and n.scope_path == ("Greeter",))
        assert greet.signature == "greet(name: string): string"


class TestDocs:
    def test_tsdoc_comment_is_parsed(self, parser):
        content = "/**\n * Does the thing.\n * @param x input\n */\nfunction f(x: number) {}\n"
        result = parser.parse("a.ts", content)
        node = next(n for n in result.nodes if n.name == "f")
        assert node.docstring == "Does the thing.\n@param x input"

    def test_tsdoc_on_an_interface(self, parser):
        content = "/** Describes a greeter. */\ninterface Greeter {\n  greet(): string;\n}\n"
        result = parser.parse("a.ts", content)
        node = next(n for n in result.nodes if n.name == "Greeter")
        assert node.docstring == "Describes a greeter."

    def test_non_tsdoc_comment_is_ignored(self, parser):
        content = "// just a regular comment\nfunction f() {}\n"
        result = parser.parse("a.ts", content)
        node = next(n for n in result.nodes if n.name == "f")
        assert node.docstring is None


class TestDefensiveBranches:
    """Match-handler fallbacks real tree-sitter queries never actually
    produce -- exercised directly since every capture group our queries
    emit is already handled."""

    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_process_definition_match_without_name_capture(self, parser):
        class FakeNode:
            type = "function_declaration"

        assert parser.process_definition_match({"def": [FakeNode()]}, b"") is None

    def test_process_reference_match_with_no_recognized_capture(self, parser):
        assert parser.process_reference_match({}, b"") is None

    def test_jsdoc_with_no_parent(self, parser):
        class FakeNode:
            parent = None

        assert parser._jsdoc(FakeNode(), b"") is None

    def test_nested_function_inside_method_finds_enclosing_construct(self, parser):
        # Exercises _nearest_enclosing_type's "found something" branch --
        # nested inside a method, so it stays a function, not a method.
        result = parser.parse("a.ts", "class C { m() { function helper() {} } }")
        node = next(n for n in result.nodes if n.name == "helper")
        assert node.kind == Kind.FUNCTION
        assert node.scope_path == ("C", "m")

    def test_arrow_signature_fallback_paths(self, parser):
        class FakeBody:
            start_byte = 5

        class FakeNodeWithBody:
            type = "arrow_function"
            start_byte = 0
            children = []

            def child_by_field_name(self, name):
                return FakeBody() if name == "body" else None

        class FakeNodeWithNothing:
            type = "arrow_function"
            start_byte = 0
            children = []

            def child_by_field_name(self, name):
                return None

        assert parser._signature(FakeNodeWithBody(), b"12345body") == "12345"
        assert parser._signature(FakeNodeWithNothing(), b"") is None

    def test_import_source_of_no_enclosing_statement_is_none(self):
        # `_enclosing` always finds a real import/export statement for
        # every capture our query produces -- exercised directly.
        from indexter.parse.typescript import _import_source

        assert _import_source(None) is None

    def test_assigned_variable_when_call_is_not_the_declarator_value(self):
        # A `require()` call our query captures is always either not inside
        # a variable_declarator or is exactly its value -- exercised
        # directly since real syntax can't produce the mismatch.
        from indexter.parse.typescript import _assigned_variable

        class FakeNode:
            def __init__(self, type_, value_field=None):
                self.type = type_
                self._value_field = value_field
                self.parent = None

            def child_by_field_name(self, name):
                return self._value_field if name == "value" else None

        call_node = FakeNode("call_expression")
        other_value = FakeNode("member_expression")
        call_node.parent = FakeNode("variable_declarator", value_field=other_value)

        assert _assigned_variable(call_node) is None


class TestReferences:
    def test_extends_and_implements_both_yield_inherits(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        inherits = [r for r in result.refs if r.ref_kind == RefKind.INHERITS]
        raw_names = {r.raw_name for r in inherits}
        assert raw_names == {"BaseHandler", "Greeter"}
        assert all(r.from_node_id.endswith("Handler#class") for r in inherits)

    def test_implements_head_matches_extends_convention(self, parser):
        result = parser.parse("a.ts", "class A extends B implements C {}\n")
        heads = {r.raw_name: r.head for r in result.refs if r.ref_kind == RefKind.INHERITS}
        assert heads == {"B": "B", "C": "C"}

    def test_multiple_implements_yield_one_ref_each(self, parser):
        result = parser.parse("a.ts", "class A implements B, C {}\n")
        inherits = [r for r in result.refs if r.ref_kind == RefKind.INHERITS]
        assert {r.raw_name for r in inherits} == {"B", "C"}

    def test_call_head(self, parser):
        result = parser.parse("a.ts", "os.path.join(a, b);\n")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.CALLS]
        assert ref.raw_name == "os.path.join"
        assert ref.head == "os"

    def test_import_specifier(self, parser):
        result = parser.parse("a.ts", 'import { X } from "./mod";\n')
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        assert ref.raw_name == "./mod"
        assert ref.imported_name == "X"
        assert ref.head == "X"

    def test_default_named_and_namespace_imports(self, parser):
        content = "import X, { a as b } from './m';\nimport * as ns from 'react';\n"
        result = parser.parse("a.ts", content)
        imports = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        by_head = {r.head: r for r in imports}
        assert by_head["X"].raw_name == "./m"
        assert by_head["X"].imported_name == "default"
        assert by_head["b"].raw_name == "./m"
        assert by_head["b"].imported_name == "a"
        assert by_head["ns"].raw_name == "react"
        assert by_head["ns"].imported_name is None

    def test_side_effect_import_has_no_head(self, parser):
        result = parser.parse("a.ts", "import './polyfill';\n")
        [ref] = result.refs
        assert ref.ref_kind == RefKind.IMPORTS
        assert ref.raw_name == "./polyfill"
        assert ref.head is None
        assert ref.imported_name is None

    def test_reexport_from(self, parser):
        content = "export { Client as C } from './client';\nexport * from './types';\n"
        result = parser.parse("a.ts", content)
        imports = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        by_raw = {r.raw_name: r for r in imports}
        assert by_raw["./client"].imported_name == "Client"
        assert by_raw["./client"].head == "C"
        assert by_raw["./types"].imported_name == "*"
        assert by_raw["./types"].head is None

    def test_local_export_is_not_an_import(self, parser):
        result = parser.parse("a.ts", "const local = 1;\nexport { local };\n")
        assert all(r.ref_kind != RefKind.IMPORTS for r in result.refs)

    def test_namespace_reexport(self, parser):
        result = parser.parse("a.ts", "export * as ns2 from './ns2mod';\n")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.IMPORTS]
        assert ref.raw_name == "./ns2mod"
        assert ref.head == "ns2"

    def test_empty_string_specifier(self, parser):
        result = parser.parse("a.ts", 'import X from "";\n')
        [ref] = result.refs
        assert ref.raw_name == ""

    def test_every_ref_origin_matches_a_node_from_the_same_parse(self, parser):
        content = (FIXTURES / "sample.ts").read_text()
        result = parser.parse("typescript/sample.ts", content)
        node_ids = {n.id for n in result.nodes}
        assert result.refs
        assert all(r.from_node_id in node_ids for r in result.refs)


class TestBuiltinDropping:
    def test_builtin_receiver_dropped(self, parser):
        result = parser.parse("a.ts", "function f() {\n    console.log('hi');\n}\n")
        assert all(r.ref_kind != RefKind.CALLS for r in result.refs)
