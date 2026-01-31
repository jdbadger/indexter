from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.models import Document, DocumentMetadata
from indexter.parser.parsers.rust import RustParser


@pytest.fixture
def rust_parser():
    """Create a RustParser instance for testing."""
    return RustParser()


@pytest.fixture
def sample_rust_document():
    """Create a sample Rust Document for testing."""
    content = """
fn simple_function() {
    println!("Hello");
}

struct SimpleStruct {
    value: i32,
}
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".rs",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.rs",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_node_content(self, rust_parser):
        source = b'fn hello() {\n    println!("hello");\n}'
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = len(source)

        content = rust_parser._get_content(node, source)
        assert content == 'fn hello() {\n    println!("hello");\n}'

    def test_should_extract_partial_content(self, rust_parser):
        source = b'// Comment\nfn hello() {\n    println!("hello");\n}'
        node = Mock(spec=Node)
        node.start_byte = 11
        node.end_byte = len(source)

        content = rust_parser._get_content(node, source)
        assert content == 'fn hello() {\n    println!("hello");\n}'

    def test_should_handle_empty_range(self, rust_parser):
        source = b"fn hello() {}"
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 0

        content = rust_parser._get_content(node, source)
        assert content == ""


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    @pytest.mark.parametrize(
        "node_type,expected",
        [
            ("function_item", "function"),
            ("struct_item", "struct"),
            ("enum_item", "enum"),
            ("trait_item", "trait"),
            ("impl_item", "impl"),
            ("mod_item", "module"),
            ("type_item", "type_alias"),
            ("const_item", "constant"),
            ("static_item", "static"),
            ("use_declaration", "import"),
        ],
    )
    def test_should_map_node_types_correctly(self, rust_parser, node_type, expected):
        node = Mock(spec=Node)
        node.type = node_type
        node.parent = None

        result = rust_parser._get_node_type(node)
        assert result == expected

    def test_should_return_method_for_impl_function(self, rust_parser):
        impl_node = Mock(spec=Node)
        impl_node.type = "impl_item"
        type_node = Mock(spec=Node)
        type_node.text = b"MyStruct"
        impl_node.child_by_field_name = Mock(return_value=type_node)
        impl_node.parent = None

        decl_list = Mock(spec=Node)
        decl_list.type = "declaration_list"
        decl_list.parent = impl_node

        node = Mock(spec=Node)
        node.type = "function_item"
        node.parent = decl_list

        node_type = rust_parser._get_node_type(node)
        assert node_type == "method"

    def test_should_return_method_for_trait_function(self, rust_parser):
        trait_node = Mock(spec=Node)
        trait_node.type = "trait_item"
        name_node = Mock(spec=Node)
        name_node.text = b"MyTrait"
        trait_node.child_by_field_name = Mock(return_value=name_node)
        trait_node.parent = None

        decl_list = Mock(spec=Node)
        decl_list.type = "declaration_list"
        decl_list.parent = trait_node

        node = Mock(spec=Node)
        node.type = "function_item"
        node.parent = decl_list

        node_type = rust_parser._get_node_type(node)
        assert node_type == "method"


class TestGetDocumentation:
    """Test the _get_documentation helper method."""

    @pytest.mark.parametrize(
        "comment_text,expected",
        [
            (b"/// This is documentation", "This is documentation"),
            (b"//! Inner doc", "Inner doc"),
        ],
    )
    def test_should_extract_doc_comments(self, rust_parser, comment_text, expected):
        comment_node = Mock(spec=Node)
        comment_node.type = "line_comment"
        comment_node.text = comment_text

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, b"")
        assert doc == expected

    def test_should_extract_multiple_line_doc_comments(self, rust_parser):
        source = b"/// Line 1\n/// Line 2\n/// Line 3\nfn hello() {}"

        comment1 = Mock(spec=Node)
        comment1.type = "line_comment"
        comment1.text = b"/// Line 1"

        comment2 = Mock(spec=Node)
        comment2.type = "line_comment"
        comment2.text = b"/// Line 2"

        comment3 = Mock(spec=Node)
        comment3.type = "line_comment"
        comment3.text = b"/// Line 3"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment1, comment2, comment3, func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, source)
        assert doc == "Line 1\nLine 2\nLine 3"

    def test_should_extract_block_doc_comment(self, rust_parser):
        source = b"/** Block doc comment */\nfn hello() {}"

        comment_node = Mock(spec=Node)
        comment_node.type = "block_comment"
        comment_node.text = b"/** Block doc comment */"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, source)
        assert doc == "Block doc comment"

    def test_should_skip_regular_comments(self, rust_parser):
        source = b"// Regular comment\nfn hello() {}"

        comment_node = Mock(spec=Node)
        comment_node.type = "line_comment"
        comment_node.text = b"// Regular comment"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, source)
        assert doc is None

    def test_should_handle_doc_comments_with_attributes(self, rust_parser):
        source = b"/// Documentation\n#[test]\nfn hello() {}"

        comment_node = Mock(spec=Node)
        comment_node.type = "line_comment"
        comment_node.text = b"/// Documentation"

        attr_node = Mock(spec=Node)
        attr_node.type = "attribute_item"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment_node, attr_node, func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, source)
        assert doc == "Documentation"

    def test_should_return_none_when_no_parent(self, rust_parser):
        node = Mock(spec=Node)
        node.parent = None

        doc = rust_parser._get_documentation(node, b"fn hello() {}")
        assert doc is None

    def test_should_return_none_when_no_doc_comments(self, rust_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, b"fn hello() {}")
        assert doc is None

    def test_should_handle_inner_doc_comments(self, rust_parser):
        source = b"//! Inner doc\nfn hello() {}"

        comment_node = Mock(spec=Node)
        comment_node.type = "line_comment"
        comment_node.text = b"//! Inner doc"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]
        func_node.parent = parent

        doc = rust_parser._get_documentation(func_node, source)
        assert doc == "Inner doc"


class TestParseBlockComment:
    """Test the _parse_block_comment helper method."""

    def test_should_parse_simple_block_comment(self, rust_parser):
        comment = "/** Simple block */"
        result = rust_parser._parse_block_comment(comment)
        assert result == "Simple block"

    def test_should_parse_multiline_block_comment(self, rust_parser):
        comment = """/**
         * Line 1
         * Line 2
         * Line 3
         */"""
        result = rust_parser._parse_block_comment(comment)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_should_handle_inner_block_comment(self, rust_parser):
        comment = "/*! Inner block */"
        result = rust_parser._parse_block_comment(comment)
        assert result == "Inner block"

    def test_should_strip_asterisks(self, rust_parser):
        comment = """/**
         * Documentation
         */"""
        result = rust_parser._parse_block_comment(comment)
        assert "*" not in result or result.count("*") == 0

    def test_should_handle_empty_block_comment(self, rust_parser):
        comment = "/**  */"
        result = rust_parser._parse_block_comment(comment)
        assert result == ""


class TestGetSignature:
    """Test the _get_signature helper method."""

    def test_should_extract_function_signature_with_body(self, rust_parser):
        source = b"fn hello(name: &str) -> String { String::from(name) }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 31

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.start_byte = 0
        func_node.end_byte = len(source)
        func_node.child_by_field_name = Mock(return_value=body_node)

        sig = rust_parser._get_signature(func_node, source)
        assert sig == "fn hello(name: &str) -> String"

    def test_should_extract_trait_method_signature_without_body(self, rust_parser):
        source = b"fn method(&self) -> i32;"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.start_byte = 0
        func_node.end_byte = len(source)
        func_node.child_by_field_name = Mock(return_value=None)

        sig = rust_parser._get_signature(func_node, source)
        assert sig == "fn method(&self) -> i32;"

    def test_should_return_none_for_non_function(self, rust_parser):
        node = Mock(spec=Node)
        node.type = "struct_item"

        sig = rust_parser._get_signature(node, b"struct Foo {}")
        assert sig is None


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_impl_type_name(self, rust_parser):
        type_node = Mock(spec=Node)
        type_node.text = b"MyStruct"

        impl_node = Mock(spec=Node)
        impl_node.type = "impl_item"
        impl_node.child_by_field_name = Mock(return_value=type_node)
        impl_node.parent = None

        decl_list = Mock(spec=Node)
        decl_list.type = "declaration_list"
        decl_list.parent = impl_node

        method_node = Mock(spec=Node)
        method_node.parent = decl_list

        parent_scope = rust_parser._get_parent_scope(method_node)
        assert parent_scope == "MyStruct"

    def test_should_return_trait_name(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyTrait"

        trait_node = Mock(spec=Node)
        trait_node.type = "trait_item"
        trait_node.child_by_field_name = Mock(return_value=name_node)
        trait_node.parent = None

        decl_list = Mock(spec=Node)
        decl_list.type = "declaration_list"
        decl_list.parent = trait_node

        method_node = Mock(spec=Node)
        method_node.parent = decl_list

        parent_scope = rust_parser._get_parent_scope(method_node)
        assert parent_scope == "MyTrait"

    def test_should_return_none_for_module_level_item(self, rust_parser):
        node = Mock(spec=Node)
        node.parent = None

        parent_scope = rust_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_traverse_through_declaration_list(self, rust_parser):
        type_node = Mock(spec=Node)
        type_node.text = b"MyStruct"

        impl_node = Mock(spec=Node)
        impl_node.type = "impl_item"
        impl_node.child_by_field_name = Mock(return_value=type_node)
        impl_node.parent = None

        decl_list = Mock(spec=Node)
        decl_list.type = "declaration_list"
        decl_list.parent = impl_node

        method_node = Mock(spec=Node)
        method_node.parent = decl_list

        parent_scope = rust_parser._get_parent_scope(method_node)
        assert parent_scope == "MyStruct"


class TestGetAttributes:
    """Test the _get_attributes helper method."""

    def test_should_extract_single_attribute(self, rust_parser):
        source = b"#[test]\nfn my_test() {}"

        attr_node = Mock(spec=Node)
        attr_node.type = "attribute_item"
        attr_node.text = b"#[test]"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [attr_node, func_node]
        func_node.parent = parent

        attrs = rust_parser._get_attributes(func_node, source)
        assert attrs == ["#[test]"]

    def test_should_extract_multiple_attributes(self, rust_parser):
        source = b"#[test]\n#[should_panic]\nfn my_test() {}"

        attr1 = Mock(spec=Node)
        attr1.type = "attribute_item"
        attr1.text = b"#[test]"

        attr2 = Mock(spec=Node)
        attr2.type = "attribute_item"
        attr2.text = b"#[should_panic]"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [attr1, attr2, func_node]
        func_node.parent = parent

        attrs = rust_parser._get_attributes(func_node, source)
        assert attrs == ["#[test]", "#[should_panic]"]

    def test_should_handle_attributes_with_doc_comments(self, rust_parser):
        source = b"/// Doc comment\n#[test]\nfn my_test() {}"

        comment_node = Mock(spec=Node)
        comment_node.type = "line_comment"
        comment_node.text = b"/// Doc comment"

        attr_node = Mock(spec=Node)
        attr_node.type = "attribute_item"
        attr_node.text = b"#[test]"

        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [comment_node, attr_node, func_node]
        func_node.parent = parent

        attrs = rust_parser._get_attributes(func_node, source)
        assert attrs == ["#[test]"]

    def test_should_return_empty_list_when_no_attributes(self, rust_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [func_node]
        func_node.parent = parent

        attrs = rust_parser._get_attributes(func_node, b"fn hello() {}")
        assert attrs == []

    def test_should_return_empty_list_when_no_parent(self, rust_parser):
        node = Mock(spec=Node)
        node.parent = None

        attrs = rust_parser._get_attributes(node, b"fn hello() {}")
        assert attrs == []


class TestGetPathName:
    """Test the _get_path_name helper method."""

    def test_should_extract_name_from_scoped_identifier(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"Read"

        path_node = Mock(spec=Node)
        path_node.type = "scoped_identifier"
        path_node.child_by_field_name = Mock(return_value=name_node)

        name = rust_parser._get_path_name(path_node)
        assert name == "Read"

    def test_should_return_text_for_simple_identifier(self, rust_parser):
        path_node = Mock(spec=Node)
        path_node.type = "identifier"
        path_node.text = b"println"
        path_node.child_by_field_name = Mock(return_value=None)

        name = rust_parser._get_path_name(path_node)
        assert name == "println"

    def test_should_handle_none_text(self, rust_parser):
        path_node = Mock(spec=Node)
        path_node.type = "scoped_identifier"
        path_node.text = None
        path_node.child_by_field_name = Mock(return_value=None)

        name = rust_parser._get_path_name(path_node)
        assert name == ""


class TestIsAsync:
    """Test the _is_async helper method."""

    def test_should_return_true_for_async_function(self, rust_parser):
        async_modifier = Mock(spec=Node)
        async_modifier.type = "async"

        modifiers_node = Mock(spec=Node)
        modifiers_node.type = "function_modifiers"
        modifiers_node.children = [async_modifier]

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = [modifiers_node]

        assert rust_parser._is_async(func_node)

    def test_should_return_false_for_sync_function(self, rust_parser):
        modifiers_node = Mock(spec=Node)
        modifiers_node.type = "function_modifiers"
        modifiers_node.children = []

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = [modifiers_node]

        assert not rust_parser._is_async(func_node)

    def test_should_return_false_for_function_without_modifiers(self, rust_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = []

        assert not rust_parser._is_async(func_node)

    def test_should_return_false_for_non_function(self, rust_parser):
        node = Mock(spec=Node)
        node.type = "struct_item"
        node.children = []

        assert not rust_parser._is_async(node)


class TestIsUnsafe:
    """Test the _is_unsafe helper method."""

    def test_should_return_true_for_unsafe_function(self, rust_parser):
        unsafe_modifier = Mock(spec=Node)
        unsafe_modifier.type = "unsafe"

        modifiers_node = Mock(spec=Node)
        modifiers_node.type = "function_modifiers"
        modifiers_node.children = [unsafe_modifier]

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = [modifiers_node]

        assert rust_parser._is_unsafe(func_node)

    def test_should_return_true_for_unsafe_direct_child(self, rust_parser):
        unsafe_node = Mock(spec=Node)
        unsafe_node.type = "unsafe"

        impl_node = Mock(spec=Node)
        impl_node.type = "impl_item"
        impl_node.children = [unsafe_node]

        assert rust_parser._is_unsafe(impl_node)

    def test_should_return_false_for_safe_function(self, rust_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = []

        assert not rust_parser._is_unsafe(func_node)

    def test_should_work_for_trait_item(self, rust_parser):
        unsafe_node = Mock(spec=Node)
        unsafe_node.type = "unsafe"

        trait_node = Mock(spec=Node)
        trait_node.type = "trait_item"
        trait_node.children = [unsafe_node]

        assert rust_parser._is_unsafe(trait_node)


class TestIsPub:
    """Test the _is_pub helper method."""

    def test_should_return_true_for_pub_item(self, rust_parser):
        vis_node = Mock(spec=Node)
        vis_node.type = "visibility_modifier"

        func_node = Mock(spec=Node)
        func_node.children = [vis_node]

        assert rust_parser._is_pub(func_node)

    def test_should_return_false_for_private_item(self, rust_parser):
        func_node = Mock(spec=Node)
        func_node.children = []

        assert not rust_parser._is_pub(func_node)


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_include_all_extra_fields(self, rust_parser):
        source = b"#[test]\npub async unsafe fn hello() {}"

        # Setup mocks for attributes
        attr_node = Mock(spec=Node)
        attr_node.type = "attribute_item"
        attr_node.text = b"#[test]"

        # Setup mocks for modifiers
        vis_node = Mock(spec=Node)
        vis_node.type = "visibility_modifier"

        async_modifier = Mock(spec=Node)
        async_modifier.type = "async"

        unsafe_modifier = Mock(spec=Node)
        unsafe_modifier.type = "unsafe"

        modifiers_node = Mock(spec=Node)
        modifiers_node.type = "function_modifiers"
        modifiers_node.children = [async_modifier, unsafe_modifier]

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = [vis_node, modifiers_node]

        parent = Mock(spec=Node)
        parent.children = [attr_node, func_node]
        func_node.parent = parent

        extra = rust_parser._get_extra(func_node, source)

        assert "attributes" in extra
        assert "#[test]" in extra["attributes"]
        assert extra["is_async"] == "true"
        assert extra["is_unsafe"] == "true"
        assert extra["is_pub"] == "true"

    def test_should_handle_no_extras(self, rust_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.children = []

        parent = Mock(spec=Node)
        parent.children = [func_node]
        parent.parent = None
        func_node.parent = parent

        extra = rust_parser._get_extra(func_node, b"fn hello() {}")

        assert extra["attributes"] == ""
        assert extra["is_async"] == "false"
        assert extra["is_unsafe"] == "false"
        assert extra["is_pub"] == "false"


# Unit tests for process_match


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, rust_parser):
        match = {}
        result = rust_parser.process_match(match, b"")
        assert result is None

    def test_should_process_function_with_name(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"my_function"

        body_node = Mock(spec=Node)
        body_node.start_byte = 20

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.start_byte = 0
        func_node.end_byte = 40
        func_node.start_point = (0, 0)
        func_node.end_point = (2, 1)
        func_node.children = []
        func_node.child_by_field_name = Mock(return_value=body_node)

        parent = Mock(spec=Node)
        parent.children = [func_node]
        parent.parent = None
        func_node.parent = parent

        match = {"def": [func_node], "name": [name_node]}
        source = b'fn my_function() {\n    println!("test");\n}'

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert "my_function" in content
        assert metadata["node_name"] == "my_function"
        assert metadata["node_type"] == "function"

    def test_should_process_use_with_alias(self, rust_parser):
        alias_node = Mock(spec=Node)
        alias_node.text = b"MyAlias"

        use_node = Mock(spec=Node)
        use_node.type = "use_declaration"
        use_node.start_byte = 0
        use_node.end_byte = 30
        use_node.start_point = (0, 0)
        use_node.end_point = (0, 30)
        use_node.children = []

        parent = Mock(spec=Node)
        parent.children = [use_node]
        parent.parent = None
        use_node.parent = parent

        match = {"def": [use_node], "alias": [alias_node]}
        source = b"use std::io::Read as MyAlias;"

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert metadata["node_name"] == "MyAlias"
        assert metadata["node_type"] == "import"

    def test_should_process_use_with_path(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"Read"

        path_node = Mock(spec=Node)
        path_node.type = "scoped_identifier"
        path_node.text = b"std::io::Read"
        path_node.child_by_field_name = Mock(return_value=name_node)

        use_node = Mock(spec=Node)
        use_node.type = "use_declaration"
        use_node.start_byte = 0
        use_node.end_byte = 20
        use_node.start_point = (0, 0)
        use_node.end_point = (0, 20)
        use_node.children = []

        parent = Mock(spec=Node)
        parent.children = [use_node]
        parent.parent = None
        use_node.parent = parent

        match = {"def": [use_node], "path": [path_node]}
        source = b"use std::io::Read;"

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert metadata["node_name"] == "Read"

    def test_should_return_none_when_no_name_or_path(self, rust_parser):
        node = Mock(spec=Node)
        node.type = "function_item"

        parent = Mock(spec=Node)
        parent.children = [node]
        parent.parent = None
        node.parent = parent

        match = {"def": [node]}
        result = rust_parser.process_match(match, b"fn () {}")
        assert result is None

    def test_should_include_documentation(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"documented_func"

        comment_node = Mock(spec=Node)
        comment_node.type = "line_comment"
        comment_node.text = b"/// This is documentation"

        body_node = Mock(spec=Node)
        body_node.start_byte = 50

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.start_byte = 26
        func_node.end_byte = 70
        func_node.start_point = (1, 0)
        func_node.end_point = (2, 1)
        func_node.children = []
        func_node.child_by_field_name = Mock(return_value=body_node)

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]
        parent.parent = None
        func_node.parent = parent

        match = {"def": [func_node], "name": [name_node]}
        source = b'/// This is documentation\nfn documented_func() {\n    println!("test");\n}'

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert metadata["documentation"] == "This is documentation"

    def test_should_include_parent_scope_for_method(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"my_method"

        type_node = Mock(spec=Node)
        type_node.text = b"MyStruct"

        impl_node = Mock(spec=Node)
        impl_node.type = "impl_item"
        impl_node.child_by_field_name = Mock(return_value=type_node)
        impl_node.parent = None

        body_node = Mock(spec=Node)
        body_node.start_byte = 30

        method_node = Mock(spec=Node)
        method_node.type = "function_item"
        method_node.start_byte = 0
        method_node.end_byte = 50
        method_node.start_point = (0, 0)
        method_node.end_point = (2, 1)
        method_node.children = []
        method_node.child_by_field_name = Mock(return_value=body_node)

        decl_list = Mock(spec=Node)
        decl_list.type = "declaration_list"
        decl_list.parent = impl_node
        decl_list.children = [method_node]
        method_node.parent = decl_list

        match = {"def": [method_node], "name": [name_node]}
        source = b'fn my_method(&self) {\n    println!("test");\n}'

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert metadata["parent_scope"] == "MyStruct"
        assert metadata["node_type"] == "method"

    def test_should_include_signature(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"add"

        body_node = Mock(spec=Node)
        body_node.start_byte = 30

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.start_byte = 0
        func_node.end_byte = 50
        func_node.start_point = (0, 0)
        func_node.end_point = (0, 50)
        func_node.children = []
        func_node.child_by_field_name = Mock(return_value=body_node)

        parent = Mock(spec=Node)
        parent.children = [func_node]
        parent.parent = None
        func_node.parent = parent

        match = {"def": [func_node], "name": [name_node]}
        source = b"fn add(a: i32, b: i32) -> i32 { a + b }"

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert metadata["signature"] == "fn add(a: i32, b: i32) -> i32"

    def test_should_include_extra_metadata(self, rust_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"test_func"

        attr_node = Mock(spec=Node)
        attr_node.type = "attribute_item"
        attr_node.text = b"#[test]"

        vis_node = Mock(spec=Node)
        vis_node.type = "visibility_modifier"

        body_node = Mock(spec=Node)
        body_node.start_byte = 30

        func_node = Mock(spec=Node)
        func_node.type = "function_item"
        func_node.start_byte = 8
        func_node.end_byte = 50
        func_node.start_point = (1, 0)
        func_node.end_point = (1, 50)
        func_node.children = [vis_node]
        func_node.child_by_field_name = Mock(return_value=body_node)

        parent = Mock(spec=Node)
        parent.children = [attr_node, func_node]
        parent.parent = None
        func_node.parent = parent

        match = {"def": [func_node], "name": [name_node]}
        source = b"#[test]\npub fn test_func() { assert!(true); }"

        result = rust_parser.process_match(match, source)
        assert result is not None
        content, metadata = result
        assert "#[test]" in metadata["extra"]["attributes"]
        assert metadata["extra"]["is_pub"] == "true"


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real Rust code."""

    def test_should_parse_simple_function(self, rust_parser):
        content = """fn hello_world() {
    println!("Hello, World!");
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        content_str, node_metadata = results[0]
        assert "hello_world" in content_str
        assert node_metadata.node_name == "hello_world"
        assert node_metadata.node_type == "function"
        assert node_metadata.language == "rust"
        assert node_metadata.start_line == 1
        assert node_metadata.end_line == 3

    def test_should_parse_function_with_documentation(self, rust_parser):
        content = """/// Adds two numbers together.
///
/// # Examples
/// ```
/// assert_eq!(add(2, 3), 5);
/// ```
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "add"
        assert "Adds two numbers together." in node_metadata.documentation
        assert "Examples" in node_metadata.documentation

    def test_should_parse_struct_definition(self, rust_parser):
        content = """pub struct Point {
    x: f64,
    y: f64,
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Point"
        assert node_metadata.node_type == "struct"
        assert node_metadata.extra["is_pub"] == "true"

    def test_should_parse_enum_definition(self, rust_parser):
        content = """pub enum Color {
    Red,
    Green,
    Blue,
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Color"
        assert node_metadata.node_type == "enum"

    def test_should_parse_trait_definition(self, rust_parser):
        content = """pub trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        # Should parse the trait and its methods
        assert len(results) >= 1
        trait_result = results[0]
        _, node_metadata = trait_result
        assert node_metadata.node_name == "Drawable"
        assert node_metadata.node_type == "trait"

    def test_should_parse_impl_block_with_methods(self, rust_parser):
        content = """struct Calculator;

impl Calculator {
    fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
    
    fn subtract(&self, a: i32, b: i32) -> i32 {
        a - b
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        # Should parse struct, impl, and two methods
        assert len(results) >= 3

        # Find methods
        methods = [r for r in results if r[1].node_type == "method"]
        assert len(methods) == 2

        method_names = {r[1].node_name for r in methods}
        assert "add" in method_names
        assert "subtract" in method_names

        # Methods should have parent scope
        for _, node_metadata in methods:
            assert node_metadata.parent_scope == "Calculator"

    def test_should_parse_async_function(self, rust_parser):
        content = """async fn fetch_data() -> Result<String, Error> {
    // Implementation
    Ok(String::from("data"))
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "fetch_data"
        assert node_metadata.extra["is_async"] == "true"

    def test_should_parse_unsafe_function(self, rust_parser):
        content = """unsafe fn dangerous_operation() {
    // Unsafe code
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.extra["is_unsafe"] == "true"

    def test_should_parse_function_with_attributes(self, rust_parser):
        content = """#[test]
#[should_panic]
fn test_panic() {
    panic!("This should panic");
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert "#[test]" in node_metadata.extra["attributes"]
        assert "#[should_panic]" in node_metadata.extra["attributes"]

    def test_should_parse_use_statements(self, rust_parser):
        content = """use std::collections::HashMap;
use std::io::Read as IoRead;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 2
        import_names = {r[1].node_name for r in results}
        assert "HashMap" in import_names
        assert "IoRead" in import_names

    def test_should_parse_constants_and_statics(self, rust_parser):
        content = """const MAX_SIZE: usize = 1024;
static GLOBAL_CONFIG: &str = "config";
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 2
        types = {r[1].node_type for r in results}
        assert "constant" in types
        assert "static" in types

    def test_should_parse_type_alias(self, rust_parser):
        content = """type Result<T> = std::result::Result<T, Error>;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Result"
        assert node_metadata.node_type == "type_alias"

    def test_should_parse_module(self, rust_parser):
        content = """mod utils {
    pub fn helper() {}
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        # Should parse module and function
        assert len(results) >= 1
        module_results = [r for r in results if r[1].node_type == "module"]
        assert len(module_results) == 1
        assert module_results[0][1].node_name == "utils"

    def test_should_parse_complex_file(self, rust_parser):
        content = """//! Main application module

use std::collections::HashMap;

/// Configuration struct
pub struct Config {
    pub name: String,
    pub value: i32,
}

impl Config {
    /// Creates a new Config instance
    pub fn new(name: String, value: i32) -> Self {
        Self { name, value }
    }
    
    /// Returns the configuration value
    pub fn get_value(&self) -> i32 {
        self.value
    }
}

/// Application trait
pub trait Application {
    fn run(&self);
}

impl Application for Config {
    fn run(&self) {
        println!("Running {}", self.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = Config::new("test".to_string(), 42);
        assert_eq!(config.get_value(), 42);
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        # Should parse multiple items
        assert len(results) >= 5

        node_types = {r[1].node_type for r in results}
        assert "import" in node_types
        assert "struct" in node_types
        assert "impl" in node_types
        assert "trait" in node_types
        assert "method" in node_types or "function" in node_types

    def test_should_handle_empty_file(self, rust_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 0

    def test_should_handle_comments_only(self, rust_parser):
        content = """// Just a comment
// Another comment
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_should_handle_generic_functions(self, rust_parser):
        content = """fn generic_func<T: Display>(item: T) -> String {
    format!("{}", item)
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "generic_func"

    def test_should_handle_lifetime_annotations(self, rust_parser):
        content = """fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "longest"

    def test_should_handle_nested_modules(self, rust_parser):
        content = """mod outer {
    pub mod inner {
        pub fn nested_func() {}
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        # Should parse both modules and the function
        assert len(results) >= 2

    def test_should_handle_tuple_struct(self, rust_parser):
        content = """pub struct Point(pub f64, pub f64);
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Point"
        assert node_metadata.node_type == "struct"

    def test_should_handle_unit_struct(self, rust_parser):
        content = """pub struct Marker;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Marker"

    def test_should_handle_enum_with_data(self, rust_parser):
        content = """pub enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Message"
        assert node_metadata.node_type == "enum"

    def test_should_handle_macro_invocations(self, rust_parser):
        # Macros aren't currently captured, but this should not break parsing
        content = """macro_rules! my_macro {
    () => {
        println!("macro");
    };
}

fn test() {
    my_macro!();
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        # Should at least parse the function
        function_results = [r for r in results if r[1].node_type == "function"]
        assert len(function_results) >= 1

    def test_should_preserve_metadata_fields(self, rust_parser):
        content = """fn test() {}"""
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/my/repo/path",
            ext=".rs",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="src/main.rs", content=content, metadata=metadata)

        results = list(rust_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/my/repo/path"
        assert node_metadata.document_path == "src/main.rs"
        assert node_metadata.language == "rust"
