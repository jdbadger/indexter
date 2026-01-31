from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.models import Document, DocumentMetadata
from indexter.parser.parsers.javascript import JavaScriptParser


@pytest.fixture
def javascript_parser():
    """Create a JavaScriptParser instance for testing."""
    return JavaScriptParser()


@pytest.fixture
def sample_javascript_document():
    """Create a sample JavaScript Document for testing."""
    content = """
function simpleFunction() {
    return 42;
}

class SimpleClass {
    method() {
        return 'hello';
    }
}
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".js",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.js",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestIsConstant:
    """Test the _is_constant helper method."""

    def test_should_return_true_for_all_uppercase(self, javascript_parser):
        assert javascript_parser._is_constant("CONSTANT")

    def test_should_return_true_for_uppercase_with_underscores(self, javascript_parser):
        assert javascript_parser._is_constant("MY_CONSTANT")
        assert javascript_parser._is_constant("API_KEY")
        assert javascript_parser._is_constant("MAX_VALUE_123")

    def test_should_return_false_for_lowercase(self, javascript_parser):
        assert not javascript_parser._is_constant("variable")
        assert not javascript_parser._is_constant("myVar")

    def test_should_return_false_for_camelCase(self, javascript_parser):
        assert not javascript_parser._is_constant("myVariable")
        assert not javascript_parser._is_constant("someFunction")

    def test_should_return_false_for_PascalCase(self, javascript_parser):
        assert not javascript_parser._is_constant("MyClass")
        assert not javascript_parser._is_constant("ComponentName")

    def test_should_return_false_for_mixed_case(self, javascript_parser):
        assert not javascript_parser._is_constant("My_Constant")
        assert not javascript_parser._is_constant("mixed_Case")

    def test_should_return_false_for_single_lowercase_letter(self, javascript_parser):
        assert not javascript_parser._is_constant("a")

    def test_should_return_true_for_single_uppercase_letter(self, javascript_parser):
        assert javascript_parser._is_constant("A")


class TestIsConstDeclaration:
    """Test the _is_const_declaration helper method."""

    def test_should_return_true_for_const_declaration(self, javascript_parser):
        const_child = Mock(spec=Node)
        const_child.type = "const"

        node = Mock(spec=Node)
        node.children = [const_child]

        assert javascript_parser._is_const_declaration(node)

    def test_should_return_false_for_let_declaration(self, javascript_parser):
        let_child = Mock(spec=Node)
        let_child.type = "let"

        node = Mock(spec=Node)
        node.children = [let_child]

        assert not javascript_parser._is_const_declaration(node)

    def test_should_return_false_for_var_declaration(self, javascript_parser):
        var_child = Mock(spec=Node)
        var_child.type = "var"

        node = Mock(spec=Node)
        node.children = [var_child]

        assert not javascript_parser._is_const_declaration(node)

    def test_should_return_false_for_empty_children(self, javascript_parser):
        node = Mock(spec=Node)
        node.children = []

        assert not javascript_parser._is_const_declaration(node)


class TestIsAsync:
    """Test the _is_async helper method."""

    def test_should_return_true_for_async_function(self, javascript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [async_child, Mock(spec=Node, type="identifier")]

        assert javascript_parser._is_async(node)

    def test_should_return_true_for_async_arrow_function(self, javascript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.children = [async_child, Mock(spec=Node, type="formal_parameters")]

        assert javascript_parser._is_async(node)

    def test_should_return_false_for_sync_function(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [
            Mock(spec=Node, type="identifier"),
            Mock(spec=Node, type="formal_parameters"),
        ]

        assert not javascript_parser._is_async(node)

    def test_should_return_false_for_non_function_node(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = []

        assert not javascript_parser._is_async(node)

    def test_should_return_true_for_async_method(self, javascript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [async_child]

        assert javascript_parser._is_async(node)


class TestIsGenerator:
    """Test the _is_generator helper method."""

    def test_should_return_true_for_generator_function_declaration(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.children = []

        assert javascript_parser._is_generator(node)

    def test_should_return_true_for_function_with_star(self, javascript_parser):
        star_child = Mock(spec=Node)
        star_child.type = "*"

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [star_child]

        assert javascript_parser._is_generator(node)

    def test_should_return_true_for_generator_method(self, javascript_parser):
        star_child = Mock(spec=Node)
        star_child.type = "*"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [star_child]

        assert javascript_parser._is_generator(node)

    def test_should_return_false_for_regular_function(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [Mock(spec=Node, type="identifier")]

        assert not javascript_parser._is_generator(node)

    def test_should_return_false_for_arrow_function(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.children = []

        assert not javascript_parser._is_generator(node)


class TestParseJSDoc:
    """Test the _parse_jsdoc helper method."""

    def test_should_parse_single_line_jsdoc(self, javascript_parser):
        comment = "/** This is a JSDoc comment */"
        result = javascript_parser._parse_jsdoc(comment)
        assert result == "This is a JSDoc comment"

    def test_should_parse_multiline_jsdoc(self, javascript_parser):
        comment = """/**
 * This is a multiline
 * JSDoc comment
 */"""
        result = javascript_parser._parse_jsdoc(comment)
        assert "This is a multiline" in result
        assert "JSDoc comment" in result

    def test_should_strip_leading_asterisks(self, javascript_parser):
        comment = """/**
 * Line one
 * Line two
 * Line three
 */"""
        result = javascript_parser._parse_jsdoc(comment)
        lines = result.split("\n")
        assert all(not line.startswith("*") for line in lines)

    def test_should_return_none_for_single_line_comment(self, javascript_parser):
        comment = "// This is not a JSDoc comment"
        result = javascript_parser._parse_jsdoc(comment)
        assert result is None

    def test_should_return_none_for_block_comment(self, javascript_parser):
        comment = "/* This is a regular block comment */"
        result = javascript_parser._parse_jsdoc(comment)
        assert result is None

    def test_should_handle_empty_jsdoc(self, javascript_parser):
        comment = "/**  */"
        result = javascript_parser._parse_jsdoc(comment)
        assert result is None

    def test_should_handle_jsdoc_with_empty_lines(self, javascript_parser):
        comment = """/**
 * First line
 *
 * Third line
 */"""
        result = javascript_parser._parse_jsdoc(comment)
        assert result is not None
        assert "First line" in result
        assert "Third line" in result

    def test_should_handle_jsdoc_with_tags(self, javascript_parser):
        comment = """/**
 * Function description
 * @param {string} name - The name
 * @returns {number} The result
 */"""
        result = javascript_parser._parse_jsdoc(comment)
        assert "Function description" in result
        assert "@param" in result
        assert "@returns" in result


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_node_content(self, javascript_parser):
        source = b"function hello() { return 'hello'; }"
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = len(source)

        content = javascript_parser._get_content(node, source)
        assert content == "function hello() { return 'hello'; }"

    def test_should_extract_partial_content(self, javascript_parser):
        source = b"// Comment\nfunction hello() { return 'hello'; }"
        node = Mock(spec=Node)
        node.start_byte = 11
        node.end_byte = len(source)

        content = javascript_parser._get_content(node, source)
        assert content == "function hello() { return 'hello'; }"

    def test_should_handle_unicode(self, javascript_parser):
        source = "const emoji = '😀';".encode()
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = len(source)

        content = javascript_parser._get_content(node, source)
        assert "😀" in content


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    def test_should_return_class_for_class_declaration(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "class_declaration"
        outer_node = actual_def

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "class"

    def test_should_return_method_for_method_definition(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "method_definition"
        outer_node = actual_def

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "method"

    def test_should_return_function_for_module_level_function(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "function_declaration"
        actual_def.parent = None
        outer_node = actual_def

        # Mock _get_parent_scope to return None for module level
        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "function"

    def test_should_return_function_for_arrow_function_module_level(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "arrow_function"
        actual_def.parent = None
        outer_node = Mock(spec=Node)
        outer_node.type = "lexical_declaration"

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "function"

    def test_should_return_constant_for_lexical_declaration(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "variable_declarator"
        outer_node = Mock(spec=Node)
        outer_node.type = "lexical_declaration"

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "constant"

    def test_should_return_import_for_import_statement(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "import_statement"
        outer_node = Mock(spec=Node)
        outer_node.type = "import_statement"

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "import"

    def test_should_return_export_for_export_statement(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "export_statement"
        outer_node = Mock(spec=Node)
        outer_node.type = "export_statement"

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "export"

    def test_should_return_function_for_generator_function(self, javascript_parser):
        actual_def = Mock(spec=Node)
        actual_def.type = "generator_function_declaration"
        actual_def.parent = None
        outer_node = actual_def

        node_type = javascript_parser._get_node_type(actual_def, outer_node)
        assert node_type == "function"


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_class_name_for_method(self, javascript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = javascript_parser._get_parent_scope(method_node)
        assert parent_scope == "MyClass"

    def test_should_return_none_for_module_level_function(self, javascript_parser):
        module_node = Mock(spec=Node)
        module_node.type = "program"
        module_node.parent = None

        function_node = Mock(spec=Node)
        function_node.parent = module_node

        parent_scope = javascript_parser._get_parent_scope(function_node)
        assert parent_scope is None

    def test_should_traverse_through_class_body(self, javascript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"OuterClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)
        class_node.parent = None

        class_body = Mock(spec=Node)
        class_body.type = "class_body"
        class_body.parent = class_node

        method_node = Mock(spec=Node)
        method_node.parent = class_body

        parent_scope = javascript_parser._get_parent_scope(method_node)
        assert parent_scope == "OuterClass"

    def test_should_return_none_when_no_class_parent(self, javascript_parser):
        node = Mock(spec=Node)
        node.parent = None

        parent_scope = javascript_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_handle_missing_class_name(self, javascript_parser):
        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=None)

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = javascript_parser._get_parent_scope(method_node)
        assert parent_scope is None


class TestGetSignature:
    """Test the _get_signature helper method."""

    def test_should_extract_function_signature(self, javascript_parser):
        source = b"function hello(name, age) { return 'hi'; }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 26

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = javascript_parser._get_signature(node, source)
        assert signature == "function hello(name, age)"

    def test_should_extract_arrow_function_signature_with_parameters(self, javascript_parser):
        source = b"(name, age) => { return 'hi'; }"

        params_node = Mock(spec=Node)
        params_node.start_byte = 0
        params_node.end_byte = 11

        arrow_child = Mock(spec=Node)
        arrow_child.type = "=>"
        arrow_child.end_byte = 14

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.start_byte = 0
        node.child_by_field_name = Mock(side_effect=lambda field: params_node if field == "parameters" else None)
        node.children = [params_node, arrow_child]

        signature = javascript_parser._get_signature(node, source)
        assert "=>" in signature
        assert signature.strip() == "(name, age) =>"

    def test_should_extract_arrow_function_signature_with_single_parameter(self, javascript_parser):
        source = b"name => { return 'hi'; }"

        param_node = Mock(spec=Node)
        param_node.start_byte = 0
        param_node.end_byte = 4

        arrow_child = Mock(spec=Node)
        arrow_child.type = "=>"
        arrow_child.end_byte = 7

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.start_byte = 0
        node.child_by_field_name = Mock(side_effect=lambda field: param_node if field == "parameter" else None)
        node.children = [param_node, arrow_child]

        signature = javascript_parser._get_signature(node, source)
        assert "=>" in signature

    def test_should_fallback_to_body_for_arrow_function(self, javascript_parser):
        source = b"(x) => x * 2"

        body_node = Mock(spec=Node)
        body_node.start_byte = 7

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.start_byte = 0
        node.child_by_field_name = Mock(side_effect=lambda field: body_node if field == "body" else None)
        node.children = []

        signature = javascript_parser._get_signature(node, source)
        assert signature == "(x) =>"

    def test_should_return_none_for_non_function_node(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"

        signature = javascript_parser._get_signature(node, b"class MyClass {}")
        assert signature is None

    def test_should_handle_method_definition(self, javascript_parser):
        source = b"methodName(param1, param2) { return true; }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 27

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = javascript_parser._get_signature(node, source)
        assert signature == "methodName(param1, param2)"

    def test_should_handle_generator_function(self, javascript_parser):
        source = b"function* generator() { yield 1; }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 22

        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = javascript_parser._get_signature(node, source)
        assert signature == "function* generator()"


class TestGetDocumentation:
    """Test the _get_documentation helper method."""

    def test_should_extract_jsdoc_from_previous_sibling(self, javascript_parser):
        source = b"/** This is JSDoc */\nfunction hello() {}"

        comment_node = Mock(spec=Node)
        comment_node.type = "comment"
        comment_node.text = b"/** This is JSDoc */"

        function_node = Mock(spec=Node)
        function_node.type = "function_declaration"

        parent_node = Mock(spec=Node)
        parent_node.children = [comment_node, function_node]

        function_node.parent = parent_node

        doc = javascript_parser._get_documentation(function_node, source)
        assert doc is not None
        assert "This is JSDoc" in doc

    def test_should_return_none_when_no_comment(self, javascript_parser):
        function_node = Mock(spec=Node)
        function_node.type = "function_declaration"

        parent_node = Mock(spec=Node)
        parent_node.children = [function_node]

        function_node.parent = parent_node

        doc = javascript_parser._get_documentation(function_node, b"function hello() {}")
        assert doc is None

    def test_should_return_none_when_no_parent(self, javascript_parser):
        node = Mock(spec=Node)
        node.parent = None

        doc = javascript_parser._get_documentation(node, b"function hello() {}")
        assert doc is None

    def test_should_ignore_non_comment_siblings(self, javascript_parser):
        other_node = Mock(spec=Node)
        other_node.type = "expression_statement"

        function_node = Mock(spec=Node)
        function_node.type = "function_declaration"

        parent_node = Mock(spec=Node)
        parent_node.children = [other_node, function_node]

        function_node.parent = parent_node

        doc = javascript_parser._get_documentation(function_node, b"x = 1;\nfunction hello() {}")
        assert doc is None

    def test_should_handle_comment_without_text(self, javascript_parser):
        comment_node = Mock(spec=Node)
        comment_node.type = "comment"
        comment_node.text = None

        function_node = Mock(spec=Node)

        parent_node = Mock(spec=Node)
        parent_node.children = [comment_node, function_node]

        function_node.parent = parent_node

        doc = javascript_parser._get_documentation(function_node, b"")
        assert doc is None


class TestGetExportName:
    """Test the _get_export_name helper method."""

    def test_should_extract_name_from_function_export(self, javascript_parser):
        source = b"export function myFunc() {}"

        name_node = Mock(spec=Node)
        name_node.text = b"myFunc"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"
        func_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export"), func_node]

        name = javascript_parser._get_export_name(export_node, source)
        assert name == "myFunc"

    def test_should_extract_name_from_class_export(self, javascript_parser):
        source = b"export class MyClass {}"

        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export"), class_node]

        name = javascript_parser._get_export_name(export_node, source)
        assert name == "MyClass"

    def test_should_extract_identifier_from_export(self, javascript_parser):
        source = b"export { myVariable }"

        identifier_node = Mock(spec=Node)
        identifier_node.type = "identifier"
        identifier_node.text = b"myVariable"

        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export"), identifier_node]

        name = javascript_parser._get_export_name(export_node, source)
        assert name == "myVariable"

    def test_should_extract_names_from_export_clause(self, javascript_parser):
        source = b"export { foo, bar }"

        name_node1 = Mock(spec=Node)
        name_node1.text = b"foo"

        name_node2 = Mock(spec=Node)
        name_node2.text = b"bar"

        spec1 = Mock(spec=Node)
        spec1.type = "export_specifier"
        spec1.child_by_field_name = Mock(return_value=name_node1)

        spec2 = Mock(spec=Node)
        spec2.type = "export_specifier"
        spec2.child_by_field_name = Mock(return_value=name_node2)

        export_clause = Mock(spec=Node)
        export_clause.type = "export_clause"
        export_clause.children = [spec1, spec2]

        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export"), export_clause]

        name = javascript_parser._get_export_name(export_node, source)
        assert name == "foo, bar"

    def test_should_return_default_for_default_export(self, javascript_parser):
        source = b"export default function() {}"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"
        func_node.child_by_field_name = Mock(return_value=None)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export"), Mock(type="default"), func_node]

        name = javascript_parser._get_export_name(export_node, source)
        assert name == "default"

    def test_should_extract_from_lexical_declaration(self, javascript_parser):
        source = b"export const myConst = 42"

        name_node = Mock(spec=Node)
        name_node.text = b"myConst"

        var_decl = Mock(spec=Node)
        var_decl.type = "variable_declarator"
        var_decl.child_by_field_name = Mock(return_value=name_node)

        lexical_decl = Mock(spec=Node)
        lexical_decl.type = "lexical_declaration"
        lexical_decl.children = [Mock(type="const"), var_decl]

        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export"), lexical_decl]

        name = javascript_parser._get_export_name(export_node, source)
        assert name == "myConst"

    def test_should_return_default_for_unknown_export(self, javascript_parser):
        export_node = Mock(spec=Node)
        export_node.children = [Mock(type="export")]

        name = javascript_parser._get_export_name(export_node, b"export default 42")
        assert name == "default"


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_return_extra_for_async_arrow_function(self, javascript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.children = [async_child]

        extra = javascript_parser._get_extra(node, b"")
        assert extra["is_async"] == "true"
        assert extra["is_generator"] == "false"
        assert extra["is_arrow"] == "true"

    def test_should_return_extra_for_generator_function(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.children = []

        extra = javascript_parser._get_extra(node, b"")
        assert extra["is_async"] == "false"
        assert extra["is_generator"] == "true"
        assert extra["is_arrow"] == "false"

    def test_should_return_extra_for_regular_function(self, javascript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [Mock(type="identifier")]

        extra = javascript_parser._get_extra(node, b"")
        assert extra["is_async"] == "false"
        assert extra["is_generator"] == "false"
        assert extra["is_arrow"] == "false"

    def test_should_return_extra_for_async_generator(self, javascript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.children = [async_child]

        extra = javascript_parser._get_extra(node, b"")
        assert extra["is_async"] == "true"
        assert extra["is_generator"] == "true"
        assert extra["is_arrow"] == "false"


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, javascript_parser):
        match = {}
        result = javascript_parser.process_match(match, b"")
        assert result is None

    def test_should_skip_lexical_declaration_inside_export(self, javascript_parser):
        export_node = Mock(spec=Node)
        export_node.type = "export_statement"

        lexical_node = Mock(spec=Node)
        lexical_node.type = "lexical_declaration"
        lexical_node.parent = export_node

        match = {"def": [lexical_node]}
        result = javascript_parser.process_match(match, b"export const X = 1")
        assert result is None

    def test_should_process_function_declaration(self, javascript_parser):
        source = b"function myFunc(a, b) { return a + b; }"

        name_node = Mock(spec=Node)
        name_node.text = b"myFunc"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"
        func_node.start_byte = 0
        func_node.end_byte = len(source)
        func_node.start_point = (0, 0)
        func_node.end_point = (0, len(source))
        func_node.parent = None
        func_node.child_by_field_name = Mock(return_value=None)
        func_node.children = []

        match = {"def": [func_node], "name": [name_node]}
        result = javascript_parser.process_match(match, source)

        assert result is not None
        content, metadata = result
        assert "myFunc" in content
        assert metadata["node_name"] == "myFunc"
        assert metadata["node_type"] == "function"

    def test_should_skip_non_constant_lexical_declaration(self, javascript_parser):
        source = b"const myVar = 42"

        name_node = Mock(spec=Node)
        name_node.text = b"myVar"

        const_child = Mock(spec=Node)
        const_child.type = "const"

        lexical_node = Mock(spec=Node)
        lexical_node.type = "lexical_declaration"
        lexical_node.parent = None
        lexical_node.children = [const_child]

        match = {"def": [lexical_node], "name": [name_node]}
        result = javascript_parser.process_match(match, source)

        # Should skip because "myVar" is not UPPER_CASE
        assert result is None

    def test_should_process_constant_lexical_declaration(self, javascript_parser):
        source = b"const MY_CONSTANT = 42"

        name_node = Mock(spec=Node)
        name_node.text = b"MY_CONSTANT"

        const_child = Mock(spec=Node)
        const_child.type = "const"

        lexical_node = Mock(spec=Node)
        lexical_node.type = "lexical_declaration"
        lexical_node.start_byte = 0
        lexical_node.end_byte = len(source)
        lexical_node.start_point = (0, 0)
        lexical_node.end_point = (0, len(source))
        lexical_node.parent = None
        lexical_node.children = [const_child]

        match = {"def": [lexical_node], "name": [name_node]}
        result = javascript_parser.process_match(match, source)

        assert result is not None
        content, metadata = result
        assert "MY_CONSTANT" in content
        assert metadata["node_name"] == "MY_CONSTANT"
        assert metadata["node_type"] == "constant"

    def test_should_process_arrow_function(self, javascript_parser):
        source = b"const myFunc = (x) => x * 2"

        name_node = Mock(spec=Node)
        name_node.text = b"myFunc"

        arrow_func = Mock(spec=Node)
        arrow_func.type = "arrow_function"
        arrow_func.start_byte = 15
        arrow_func.end_byte = 27
        arrow_func.start_point = (0, 15)
        arrow_func.end_point = (0, 27)
        arrow_func.parent = None
        arrow_func.children = []
        arrow_func.child_by_field_name = Mock(return_value=None)

        lexical_node = Mock(spec=Node)
        lexical_node.type = "lexical_declaration"
        lexical_node.start_byte = 0
        lexical_node.end_byte = len(source)
        lexical_node.start_point = (0, 0)
        lexical_node.end_point = (0, len(source))
        lexical_node.parent = None

        match = {"def": [lexical_node], "name": [name_node], "arrow_func": [arrow_func]}
        result = javascript_parser.process_match(match, source)

        assert result is not None
        content, metadata = result
        assert metadata["node_name"] == "myFunc"
        assert metadata["extra"]["is_arrow"] == "true"

    def test_should_process_import_statement(self, javascript_parser):
        source = b"import { foo } from './module'"

        source_node = Mock(spec=Node)
        source_node.text = b"'./module'"

        parent_node = Mock(spec=Node)
        parent_node.type = "program"
        parent_node.children = []
        parent_node.parent = None

        import_node = Mock(spec=Node)
        import_node.type = "import_statement"
        import_node.start_byte = 0
        import_node.end_byte = len(source)
        import_node.start_point = (0, 0)
        import_node.end_point = (0, len(source))
        import_node.parent = parent_node

        match = {"def": [import_node], "source": [source_node]}
        result = javascript_parser.process_match(match, source)

        assert result is not None
        content, metadata = result
        assert metadata["node_name"] == "./module"
        assert metadata["node_type"] == "import"

    def test_should_process_class_declaration(self, javascript_parser):
        source = b"class MyClass { constructor() {} }"

        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        parent_node = Mock(spec=Node)
        parent_node.type = "program"
        parent_node.children = []
        parent_node.parent = None

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.start_byte = 0
        class_node.end_byte = len(source)
        class_node.start_point = (0, 0)
        class_node.end_point = (0, len(source))
        class_node.children = []
        class_node.child_by_field_name = Mock(return_value=None)
        class_node.parent = parent_node

        match = {"def": [class_node], "name": [name_node]}
        result = javascript_parser.process_match(match, source)

        assert result is not None
        content, metadata = result
        assert metadata["node_name"] == "MyClass"
        assert metadata["node_type"] == "class"


# Integration tests


class TestJavaScriptParserIntegration:
    """Integration tests for the JavaScript parser."""

    def test_should_parse_simple_function(self, javascript_parser):
        content = """
function add(a, b) {
    return a + b;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        content_str, node_metadata = results[0]
        assert "add" in content_str
        assert node_metadata.node_name == "add"
        assert node_metadata.node_type == "function"

    def test_should_parse_class_with_methods(self, javascript_parser):
        content = """
class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should find class and methods
        assert len(results) >= 3

        class_result = next((r for r in results if r[1].node_name == "Calculator"), None)
        assert class_result is not None
        assert class_result[1].node_type == "class"

        add_method = next((r for r in results if r[1].node_name == "add"), None)
        assert add_method is not None
        assert add_method[1].node_type == "method"
        assert add_method[1].parent_scope == "Calculator"

    def test_should_parse_arrow_functions(self, javascript_parser):
        content = """
const square = (x) => x * x;
const greet = name => `Hello, ${name}!`;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) >= 2

        square_func = next((r for r in results if r[1].node_name == "square"), None)
        assert square_func is not None
        assert square_func[1].extra["is_arrow"] == "true"

    def test_should_parse_async_functions(self, javascript_parser):
        content = """
async function fetchData() {
    const response = await fetch('/api');
    return response.json();
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        func_result = results[0]
        assert func_result[1].node_name == "fetchData"
        assert func_result[1].extra["is_async"] == "true"

    def test_should_parse_generator_functions(self, javascript_parser):
        content = """
function* idGenerator() {
    let id = 0;
    while (true) {
        yield id++;
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        func_result = results[0]
        assert func_result[1].node_name == "idGenerator"
        assert func_result[1].extra["is_generator"] == "true"

    def test_should_parse_constants(self, javascript_parser):
        content = """
const API_KEY = 'secret123';
const MAX_RETRIES = 3;
const myVariable = 42;  // Should be skipped
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should find API_KEY and MAX_RETRIES, but not myVariable
        constant_names = [r[1].node_name for r in results]
        assert "API_KEY" in constant_names
        assert "MAX_RETRIES" in constant_names
        assert "myVariable" not in constant_names

    def test_should_parse_import_statements(self, javascript_parser):
        content = """
import React from 'react';
import { useState, useEffect } from 'react';
import './styles.css';
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should find import statements
        import_results = [r for r in results if r[1].node_type == "import"]
        assert len(import_results) >= 3

    def test_should_parse_export_statements(self, javascript_parser):
        content = """
export function myFunction() {
    return 'exported';
}

export class MyClass {}

export const MY_CONST = 42;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should find exports
        export_results = [r for r in results if r[1].node_type == "export"]
        assert len(export_results) >= 3

    def test_should_parse_jsdoc_comments(self, javascript_parser):
        content = """
/**
 * Adds two numbers together
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        func_result = results[0]
        assert func_result[1].documentation is not None
        assert "Adds two numbers together" in func_result[1].documentation
        assert "@param" in func_result[1].documentation

    def test_should_extract_function_signatures(self, javascript_parser):
        content = """
function regularFunc(a, b, c) {
    return a + b + c;
}

const arrowFunc = (x, y) => x * y;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        regular_func = next((r for r in results if r[1].node_name == "regularFunc"), None)
        assert regular_func is not None
        assert regular_func[1].signature is not None
        assert "a, b, c" in regular_func[1].signature

    def test_should_handle_nested_functions(self, javascript_parser):
        content = """
function outer() {
    function inner() {
        return 42;
    }
    return inner();
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should find both functions
        function_names = [r[1].node_name for r in results]
        assert "outer" in function_names
        assert "inner" in function_names

    def test_should_handle_complex_class_structure(self, javascript_parser):
        content = """
/**
 * A calculator class
 */
class Calculator {
    constructor() {
        this.result = 0;
    }
    
    /**
     * Add numbers
     */
    add(a, b) {
        return a + b;
    }
    
    async fetchValue() {
        return await fetch('/value');
    }
    
    *generateSequence() {
        yield 1;
        yield 2;
    }
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Find class
        class_result = next((r for r in results if r[1].node_name == "Calculator"), None)
        assert class_result is not None
        assert class_result[1].documentation is not None

        # Find methods
        add_method = next((r for r in results if r[1].node_name == "add"), None)
        assert add_method is not None
        assert add_method[1].parent_scope == "Calculator"
        assert add_method[1].documentation is not None

        # Check async method
        async_method = next((r for r in results if r[1].node_name == "fetchValue"), None)
        assert async_method is not None
        assert async_method[1].extra["is_async"] == "true"

        # Check generator method
        generator_method = next((r for r in results if r[1].node_name == "generateSequence"), None)
        assert generator_method is not None
        assert generator_method[1].extra["is_generator"] == "true"

    def test_should_handle_empty_file(self, javascript_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) == 0

    def test_should_handle_comments_only(self, javascript_parser):
        content = """
// This is a comment
/* This is a block comment */
/**
 * This is a JSDoc comment
 */
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="comments.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should not parse standalone comments
        assert len(results) == 0

    def test_should_include_repo_metadata(self, javascript_parser):
        content = """
function test() {
    return 'test';
}
"""
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/home/user/projects/my-repo",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="src/test.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        _, node_metadata = results[0]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/home/user/projects/my-repo"
        assert node_metadata.document_path == "src/test.js"

    def test_should_handle_unicode_content(self, javascript_parser):
        content = """
function greet() {
    return '你好世界 🌍';
}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content.encode("utf-8")),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        content_str, _ = results[0]
        assert "你好世界" in content_str
        assert "🌍" in content_str

    def test_should_handle_mixed_exports_and_declarations(self, javascript_parser):
        content = """
export function exportedFunc() {}

function regularFunc() {}

export class ExportedClass {}

class RegularClass {}

export const EXPORTED_CONST = 1;
const REGULAR_CONST = 2;
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="mixed.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        # Should find all declarations
        node_names = [r[1].node_name for r in results]
        assert "exportedFunc" in node_names or any("export" in r[1].node_type for r in results)
        assert "regularFunc" in node_names
        assert "RegularClass" in node_names

    def test_should_preserve_line_numbers(self, javascript_parser):
        content = """// Line 1
// Line 2
function myFunc() {  // Line 3
    return 42;       // Line 4
}                    // Line 5
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="lines.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        assert len(results) > 0
        _, node_metadata = results[0]
        assert node_metadata.start_line == 3
        assert node_metadata.end_line == 5


class TestJavaScriptParserEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_malformed_javascript(self, javascript_parser):
        """Tree-sitter should still parse and extract what it can."""
        content = """
function incomplete(
    // Missing closing brace and parameter list
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="malformed.js", content=content, metadata=metadata)

        # Should not raise an exception
        results = list(javascript_parser.parse(document))
        # May or may not find anything depending on tree-sitter's error recovery
        assert isinstance(results, list)

    def test_should_handle_very_long_function_names(self, javascript_parser):
        long_name = "a" * 1000
        content = f"""
function {long_name}() {{
    return 42;
}}
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="long.js", content=content, metadata=metadata)

        results = list(javascript_parser.parse(document))

        if len(results) > 0:
            assert len(results[0][1].node_name) == 1000

    def test_should_handle_deeply_nested_structures(self, javascript_parser):
        # Create deeply nested arrow functions
        content = "const a = () => () => () => () => () => 42;"

        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="nested.js", content=content, metadata=metadata)

        # Should handle without stack overflow
        results = list(javascript_parser.parse(document))
        assert isinstance(results, list)

    def test_should_not_create_circular_references_in_mocks(self, javascript_parser):
        """Ensure mocks don't create infinite loops."""
        source = b"function test() {}"

        # Create a mock that could potentially cause issues
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.parent = None
        node.children = []
        node.start_byte = 0
        node.end_byte = len(source)
        node.start_point = (0, 0)
        node.end_point = (0, len(source))

        # This should not cause infinite recursion
        parent_scope = javascript_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_handle_export_default_arrow_function(self, javascript_parser):
        content = """
export default () => {
    return 'default export';
};
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/test",
            ext=".js",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="export-arrow.js", content=content, metadata=metadata)

        # Should handle without error
        results = list(javascript_parser.parse(document))
        assert isinstance(results, list)
