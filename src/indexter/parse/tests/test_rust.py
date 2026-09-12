from pathlib import Path

import pytest

from indexter.parse.models import Kind, RefKind
from indexter.parse.rust import RustParser

FIXTURES = Path(__file__).parent / "fixtures" / "rust"


@pytest.fixture
def parser():
    return RustParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.rs").read_text()
        result = parser.parse("rust/sample.rs", content)
        kinds = {n.name: n.kind for n in result.nodes if n.name}
        assert kinds["Foo"] == Kind.STRUCT
        assert kinds["Greet"] == Kind.TRAIT
        assert kinds["new"] == Kind.METHOD
        assert kinds["greet"] == Kind.METHOD

    def test_no_impl_or_use_nodes(self, parser):
        content = (FIXTURES / "sample.rs").read_text()
        result = parser.parse("rust/sample.rs", content)
        assert all(n.kind not in ("impl", "import", "module") for n in result.nodes)

    def test_byte_ranges_slice_back_to_exact_source(self, parser):
        content = (FIXTURES / "sample.rs").read_text()
        result = parser.parse("rust/sample.rs", content)
        source_bytes = content.encode()
        node = next(n for n in result.nodes if n.name == "new")
        text = source_bytes[node.start_byte : node.end_byte].decode()
        assert text.startswith("pub fn new")

    def test_empty_file(self, parser):
        result = parser.parse("a.rs", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]
        assert result.errors == []

    def test_const_and_static_are_constants(self, parser):
        result = parser.parse("a.rs", "const MAX: i32 = 5;\nstatic NAME: &str = \"x\";\n")
        kinds = {n.name: n.kind for n in result.nodes if n.name}
        assert kinds["MAX"] == Kind.CONSTANT
        assert kinds["NAME"] == Kind.CONSTANT

    def test_type_alias(self, parser):
        result = parser.parse("a.rs", "type Alias = Vec<i32>;\n")
        node = next(n for n in result.nodes if n.name == "Alias")
        assert node.kind == Kind.TYPE_ALIAS

    def test_enum(self, parser):
        result = parser.parse("a.rs", "enum Status { Active, Inactive }\n")
        node = next(n for n in result.nodes if n.name == "Status")
        assert node.kind == Kind.ENUM

    def test_free_function_is_function_kind(self, parser):
        result = parser.parse("a.rs", "fn helper() {}\n")
        node = next(n for n in result.nodes if n.name == "helper")
        assert node.kind == Kind.FUNCTION


class TestCollisions:
    def test_trait_impls_of_same_type_are_distinct(self, parser):
        content = (FIXTURES / "sample.rs").read_text()
        result = parser.parse("rust/sample.rs", content)
        fmt_methods = [n for n in result.nodes if n.name == "fmt"]
        assert len(fmt_methods) == 2
        scopes = {n.scope_path for n in fmt_methods}
        assert scopes == {("Foo<Display>",), ("Foo<Debug>",)}
        assert fmt_methods[0].id != fmt_methods[1].id
        assert "~" not in fmt_methods[0].id
        assert "~" not in fmt_methods[1].id

    def test_inherent_impl_has_no_trait_suffix(self, parser):
        content = (FIXTURES / "sample.rs").read_text()
        result = parser.parse("rust/sample.rs", content)
        new_method = next(n for n in result.nodes if n.name == "new")
        assert new_method.scope_path == ("Foo",)

    def test_trait_path_reduced_to_last_segment(self, parser):
        result = parser.parse(
            "a.rs",
            "struct Foo;\nimpl std::fmt::Display for Foo { fn fmt(&self) {} }\n",
        )
        method = next(n for n in result.nodes if n.name == "fmt")
        assert method.scope_path == ("Foo<Display>",)


class TestDocComments:
    def test_line_doc_comment_with_attribute(self, parser):
        content = "/// A doc comment.\n/// Second line.\n#[derive(Debug)]\npub fn documented() {}\n"
        result = parser.parse("a.rs", content)
        node = next(n for n in result.nodes if n.name == "documented")
        assert node.docstring == "A doc comment.\nSecond line."

    def test_block_doc_comment(self, parser):
        content = "/** block doc */\npub fn blockdoc() {}\n"
        result = parser.parse("a.rs", content)
        node = next(n for n in result.nodes if n.name == "blockdoc")
        assert node.docstring == "block doc"

    def test_regular_comment_is_not_a_doc_comment(self, parser):
        content = "// just a comment\npub fn undocumented() {}\n"
        result = parser.parse("a.rs", content)
        node = next(n for n in result.nodes if n.name == "undocumented")
        assert node.docstring is None

    def test_trait_default_method_scoped_to_trait(self, parser):
        result = parser.parse("a.rs", "trait Greet { fn hello(&self) { println!(\"hi\"); } }\n")
        node = next(n for n in result.nodes if n.name == "hello")
        assert node.scope_path == ("Greet",)
        assert node.kind == Kind.METHOD


class TestDefensiveBranches:
    """Match-handler and helper fallbacks real tree-sitter queries and
    parses never actually produce -- exercised directly."""

    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_process_definition_match_without_name_capture(self, parser):
        class FakeNode:
            type = "function_item"

        assert parser.process_definition_match({"def": [FakeNode()]}, b"") is None

    def test_process_reference_match_with_no_recognized_capture(self, parser):
        assert parser.process_reference_match({}, b"") is None

    def test_scope_segment_impl_with_no_type_text(self, parser):
        class FakeTypeNode:
            text = None

        class FakeImpl:
            type = "impl_item"

            def child_by_field_name(self, name):
                return FakeTypeNode() if name == "type" else None

        assert parser.scope_segment(FakeImpl()) is None

    def test_doc_comment_with_no_parent(self, parser):
        class FakeNode:
            parent = None

        assert parser._doc_comment(FakeNode()) is None

    def test_doc_comment_node_not_found_among_siblings(self, parser):
        class FakeParent:
            children = []

        class FakeNode:
            parent = FakeParent()

        assert parser._doc_comment(FakeNode()) is None

    def test_regular_block_comment_is_not_a_doc_comment(self, parser):
        content = "/* not a doc comment */\npub fn f() {}\n"
        result = parser.parse("a.rs", content)
        node = next(n for n in result.nodes if n.name == "f")
        assert node.docstring is None

    def test_group_item_binding_use_as_clause_missing_fields(self):
        from indexter.parse.rust import _group_item_binding

        class FakeNode:
            type = "use_as_clause"

            def child_by_field_name(self, name):
                return None

        assert _group_item_binding(FakeNode(), "std") is None

    def test_group_item_binding_unrecognized_inner_type(self):
        from indexter.parse.rust import _group_item_binding

        class FakeItem:
            type = "scoped_use_list"  # a nested group -- not supported

        assert _group_item_binding(FakeItem(), "std") is None

    def test_use_wildcard_with_no_children(self, parser):
        class FakeNode:
            type = "use_wildcard"
            child_count = 0
            children = []
            start_byte = 0
            start_point = (0, 0)

        ref = parser.process_reference_match({"use_wildcard": [FakeNode()]}, b"")
        assert ref.raw_name == ""

    def test_use_aliased_missing_fields_yields_no_ref(self, parser):
        class FakeNode:
            type = "use_as_clause"

            def child_by_field_name(self, name):
                return None

        assert parser.process_reference_match({"use_aliased": [FakeNode()]}, b"") is None

    def test_use_group_item_with_unrecognized_binding_yields_no_ref(self, parser):
        class FakePrefix:
            text = b"std"

        class FakeItem:
            type = "scoped_use_list"  # a nested group -- not supported

        match = {"use_group_item": [FakeItem()], "use_group_prefix": [FakePrefix()]}
        assert parser.process_reference_match(match, b"") is None


class TestReferences:
    def test_self_field_chain_head(self, parser):
        result = parser.parse("a.rs", "fn f() { self.x.y(); }\n")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.CALLS]
        assert ref.raw_name == "self.x.y"
        assert ref.head == "self"

    def test_scoped_identifier_call_head(self, parser):
        result = parser.parse("a.rs", "fn f() { Self::new(); }\n")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.CALLS]
        assert ref.raw_name == "Self::new"
        assert ref.head == "Self"

    def test_use_declaration_is_import(self, parser):
        result = parser.parse("a.rs", "use std::fmt;\n")
        [ref] = result.refs
        assert ref.ref_kind == RefKind.IMPORTS
        assert ref.raw_name == "std"
        assert ref.imported_name == "fmt"
        assert ref.head == "fmt"

    def test_use_crate_path_head(self, parser):
        result = parser.parse("a.rs", "use crate::helpers::assist;\n")
        [ref] = result.refs
        assert ref.raw_name == "crate::helpers"
        assert ref.imported_name == "assist"
        assert ref.head == "assist"

    def test_use_single_segment_has_no_imported_name(self, parser):
        result = parser.parse("a.rs", "use serde;\n")
        [ref] = result.refs
        assert ref.raw_name == "serde"
        assert ref.imported_name is None
        assert ref.head == "serde"

    def test_use_aliased_yields_alias_head(self, parser):
        result = parser.parse("a.rs", "use std::collections::HashMap as Map;\n")
        [ref] = result.refs
        assert ref.raw_name == "std::collections"
        assert ref.imported_name == "HashMap"
        assert ref.head == "Map"

    def test_use_wildcard(self, parser):
        result = parser.parse("a.rs", "use std::collections::*;\n")
        [ref] = result.refs
        assert ref.raw_name == "std::collections"
        assert ref.imported_name == "*"
        assert ref.head is None

    def test_braced_use_list_yields_one_ref_per_item(self, parser):
        result = parser.parse("a.rs", "use std::{io, fmt::Display as Show};\n")
        refs = {r.imported_name: r for r in result.refs}
        assert refs["io"].raw_name == "std"
        assert refs["io"].head == "io"
        assert refs["Display"].raw_name == "std::fmt"
        assert refs["Display"].head == "Show"

    def test_trait_impl_yields_inherits(self, parser):
        result = parser.parse("a.rs", "struct Foo;\nimpl Display for Foo {}\n")
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.INHERITS]
        assert ref.raw_name == "Display"
        assert ref.head == "Display"
        assert ref.for_type == "Foo"

    def test_inherent_impl_yields_no_inherits(self, parser):
        result = parser.parse("a.rs", "struct Foo;\nimpl Foo { fn new() -> Foo { Foo } }\n")
        assert all(r.ref_kind != RefKind.INHERITS for r in result.refs)

    def test_every_ref_origin_matches_a_node_from_the_same_parse(self, parser):
        content = (FIXTURES / "sample.rs").read_text()
        result = parser.parse("rust/sample.rs", content)
        node_ids = {n.id for n in result.nodes}
        assert result.refs
        assert all(r.from_node_id in node_ids for r in result.refs)


class TestBuiltinDropping:
    def test_prelude_constructors_dropped(self, parser):
        content = "fn f() {\n    Some(1);\n    Vec::new();\n}\n"
        result = parser.parse("a.rs", content)
        assert all(r.ref_kind != RefKind.CALLS for r in result.refs)
