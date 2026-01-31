from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.models import Document, DocumentMetadata
from indexter.parser.parsers.typescript import TypeScriptParser


@pytest.fixture
def typescript_parser():
    """Create a TypeScriptParser instance for testing."""
    return TypeScriptParser()


@pytest.fixture
def sample_typescript_document():
    """Create a sample TypeScript Document for testing."""
    content = """
function simpleFunction() {
    return true;
}

class SimpleClass {
    method() {}
}
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".ts",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.ts",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestIsConstant:
    """Test the _is_constant helper method."""

    def test_should_return_true_for_all_uppercase(self, typescript_parser):
        assert typescript_parser._is_constant("CONSTANT")

    def test_should_return_true_for_uppercase_with_underscores(self, typescript_parser):
        assert typescript_parser._is_constant("MY_CONSTANT")
        assert typescript_parser._is_constant("API_KEY")
        assert typescript_parser._is_constant("MAX_VALUE_123")

    def test_should_return_false_for_lowercase(self, typescript_parser):
        assert not typescript_parser._is_constant("variable")
        assert not typescript_parser._is_constant("myVar")

    def test_should_return_false_for_mixed_case(self, typescript_parser):
        assert not typescript_parser._is_constant("MyClass")
        assert not typescript_parser._is_constant("myVariable")
        assert not typescript_parser._is_constant("My_Constant")

    def test_should_return_false_for_single_lowercase_letter(self, typescript_parser):
        assert not typescript_parser._is_constant("a")

    def test_should_return_true_for_single_uppercase_letter(self, typescript_parser):
        assert typescript_parser._is_constant("A")


class TestIsConstDeclaration:
    """Test the _is_const_declaration helper method."""

    def test_should_return_true_when_const_keyword_present(self, typescript_parser):
        const_child = Mock(spec=Node)
        const_child.type = "const"

        other_child = Mock(spec=Node)
        other_child.type = "variable_declarator"

        node = Mock(spec=Node)
        node.type = "lexical_declaration"
        node.children = [const_child, other_child]

        assert typescript_parser._is_const_declaration(node)

    def test_should_return_false_when_let_keyword_present(self, typescript_parser):
        let_child = Mock(spec=Node)
        let_child.type = "let"

        other_child = Mock(spec=Node)
        other_child.type = "variable_declarator"

        node = Mock(spec=Node)
        node.type = "lexical_declaration"
        node.children = [let_child, other_child]

        assert not typescript_parser._is_const_declaration(node)

    def test_should_return_false_when_no_const_keyword(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "lexical_declaration"
        node.children = [
            Mock(spec=Node, type="variable_declarator"),
            Mock(spec=Node, type="identifier"),
        ]

        assert not typescript_parser._is_const_declaration(node)

    def test_should_return_false_for_empty_children(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "lexical_declaration"
        node.children = []

        assert not typescript_parser._is_const_declaration(node)


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_node_content(self, typescript_parser):
        source = b"function hello() { console.log('hello'); }"
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 42

        content = typescript_parser._get_content(node, source)
        assert content == "function hello() { console.log('hello'); }"

    def test_should_extract_partial_content(self, typescript_parser):
        source = b"// Comment\nfunction hello() { return true; }"
        node = Mock(spec=Node)
        node.start_byte = 11
        node.end_byte = 44

        content = typescript_parser._get_content(node, source)
        assert content == "function hello() { return true; }"


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    def test_should_return_class_for_class_declaration(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "class"

    def test_should_return_class_for_abstract_class(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "abstract_class_declaration"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "class"

    def test_should_return_interface_for_interface_declaration(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "interface_declaration"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "interface"

    def test_should_return_type_alias_for_type_alias_declaration(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "type_alias_declaration"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "type_alias"

    def test_should_return_enum_for_enum_declaration(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "enum_declaration"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "enum"

    def test_should_return_method_for_method_definition(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "method_definition"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "method"

    def test_should_return_method_for_method_signature(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "method_signature"

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "method"

    def test_should_return_function_for_module_level_function(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.parent = None

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "function"

    def test_should_return_method_for_class_function(self, typescript_parser):
        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.parent = None

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.parent = class_node

        node_type = typescript_parser._get_node_type(node, node)
        assert node_type == "method"

    def test_should_return_function_for_arrow_function_at_module_level(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.parent = None

        outer_node = Mock(spec=Node)
        outer_node.type = "lexical_declaration"

        node_type = typescript_parser._get_node_type(node, outer_node)
        assert node_type == "function"

    def test_should_return_constant_for_non_arrow_lexical_declaration(self, typescript_parser):
        outer_node = Mock(spec=Node)
        outer_node.type = "lexical_declaration"

        actual_node = Mock(spec=Node)
        actual_node.type = "identifier"

        node_type = typescript_parser._get_node_type(actual_node, outer_node)
        assert node_type == "constant"

    def test_should_return_import_for_import_statement(self, typescript_parser):
        outer_node = Mock(spec=Node)
        outer_node.type = "import_statement"

        actual_node = Mock(spec=Node)
        actual_node.type = "import_statement"

        node_type = typescript_parser._get_node_type(actual_node, outer_node)
        assert node_type == "import"

    def test_should_return_export_for_export_statement(self, typescript_parser):
        outer_node = Mock(spec=Node)
        outer_node.type = "export_statement"

        actual_node = Mock(spec=Node)
        actual_node.type = "export_statement"

        node_type = typescript_parser._get_node_type(actual_node, outer_node)
        assert node_type == "export"

    def test_should_return_original_type_for_unknown(self, typescript_parser):
        outer_node = Mock(spec=Node)
        outer_node.type = "unknown_type"

        actual_node = Mock(spec=Node)
        actual_node.type = "unknown_type"

        node_type = typescript_parser._get_node_type(actual_node, outer_node)
        assert node_type == "unknown_type"


class TestGetVisibility:
    """Test the _get_visibility helper method."""

    def test_should_return_public_for_public_modifier(self, typescript_parser):
        modifier = Mock(spec=Node)
        modifier.type = "accessibility_modifier"
        modifier.text = b"public"

        node = Mock(spec=Node)
        node.children = [modifier]

        visibility = typescript_parser._get_visibility(node)
        assert visibility == "public"

    def test_should_return_private_for_private_modifier(self, typescript_parser):
        modifier = Mock(spec=Node)
        modifier.type = "accessibility_modifier"
        modifier.text = b"private"

        node = Mock(spec=Node)
        node.children = [modifier]

        visibility = typescript_parser._get_visibility(node)
        assert visibility == "private"

    def test_should_return_protected_for_protected_modifier(self, typescript_parser):
        modifier = Mock(spec=Node)
        modifier.type = "accessibility_modifier"
        modifier.text = b"protected"

        node = Mock(spec=Node)
        node.children = [modifier]

        visibility = typescript_parser._get_visibility(node)
        assert visibility == "protected"

    def test_should_return_none_when_no_accessibility_modifier(self, typescript_parser):
        node = Mock(spec=Node)
        node.children = [
            Mock(spec=Node, type="identifier"),
            Mock(spec=Node, type="parameters"),
        ]

        visibility = typescript_parser._get_visibility(node)
        assert visibility is None

    def test_should_return_none_for_empty_children(self, typescript_parser):
        node = Mock(spec=Node)
        node.children = []

        visibility = typescript_parser._get_visibility(node)
        assert visibility is None


class TestParseTsdoc:
    """Test the _parse_tsdoc helper method."""

    def test_should_parse_jsdoc_comment(self, typescript_parser):
        comment = "/**\n * This is a JSDoc comment\n */"
        result = typescript_parser._parse_tsdoc(comment)
        assert result == "This is a JSDoc comment"

    def test_should_parse_multiline_jsdoc(self, typescript_parser):
        comment = "/**\n * First line\n * Second line\n * Third line\n */"
        result = typescript_parser._parse_tsdoc(comment)
        assert result == "First line\nSecond line\nThird line"

    def test_should_strip_leading_asterisks(self, typescript_parser):
        comment = "/**\n * Line with asterisk\n *Line without space\n */"
        result = typescript_parser._parse_tsdoc(comment)
        assert "Line with asterisk" in result
        assert "Line without space" in result

    def test_should_return_none_for_single_line_comment(self, typescript_parser):
        comment = "// Single line comment"
        result = typescript_parser._parse_tsdoc(comment)
        assert result is None

    def test_should_return_none_for_block_comment_without_double_star(self, typescript_parser):
        comment = "/* Regular block comment */"
        result = typescript_parser._parse_tsdoc(comment)
        assert result is None

    def test_should_handle_empty_jsdoc(self, typescript_parser):
        comment = "/**\n *\n */"
        result = typescript_parser._parse_tsdoc(comment)
        assert result is None or result == ""

    def test_should_handle_jsdoc_with_tags(self, typescript_parser):
        comment = "/**\n * Function description\n * @param x The parameter\n * @returns The result\n */"
        result = typescript_parser._parse_tsdoc(comment)
        assert "Function description" in result
        assert "@param" in result
        assert "@returns" in result


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_class_name_for_method(self, typescript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)
        class_node.parent = None

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = typescript_parser._get_parent_scope(method_node)
        assert parent_scope == "MyClass"

    def test_should_return_class_name_for_abstract_class(self, typescript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"AbstractClass"

        class_node = Mock(spec=Node)
        class_node.type = "abstract_class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)
        class_node.parent = None

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = typescript_parser._get_parent_scope(method_node)
        assert parent_scope == "AbstractClass"

    def test_should_return_interface_name_for_method_signature(self, typescript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyInterface"

        interface_node = Mock(spec=Node)
        interface_node.type = "interface_declaration"
        interface_node.child_by_field_name = Mock(return_value=name_node)
        interface_node.parent = None

        method_node = Mock(spec=Node)
        method_node.parent = interface_node

        parent_scope = typescript_parser._get_parent_scope(method_node)
        assert parent_scope == "MyInterface"

    def test_should_return_none_for_module_level_function(self, typescript_parser):
        node = Mock(spec=Node)
        node.parent = None

        parent_scope = typescript_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_traverse_through_class_body(self, typescript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"OuterClass"

        outer_class = Mock(spec=Node)
        outer_class.type = "class_declaration"
        outer_class.child_by_field_name = Mock(return_value=name_node)
        outer_class.parent = None

        class_body = Mock(spec=Node)
        class_body.type = "class_body"
        class_body.parent = outer_class

        method_node = Mock(spec=Node)
        method_node.parent = class_body

        parent_scope = typescript_parser._get_parent_scope(method_node)
        assert parent_scope == "OuterClass"

    def test_should_traverse_through_interface_body(self, typescript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyInterface"

        interface_node = Mock(spec=Node)
        interface_node.type = "interface_declaration"
        interface_node.child_by_field_name = Mock(return_value=name_node)
        interface_node.parent = None

        interface_body = Mock(spec=Node)
        interface_body.type = "interface_body"
        interface_body.parent = interface_node

        method_node = Mock(spec=Node)
        method_node.parent = interface_body

        parent_scope = typescript_parser._get_parent_scope(method_node)
        assert parent_scope == "MyInterface"

    def test_should_return_none_when_no_name(self, typescript_parser):
        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=None)
        class_node.parent = None

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = typescript_parser._get_parent_scope(method_node)
        assert parent_scope is None


class TestGetSignature:
    """Test the _get_signature helper method."""

    def test_should_extract_function_signature(self, typescript_parser):
        source = b"function myFunc(arg1: string, arg2: number): boolean { return true; }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 53

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = typescript_parser._get_signature(node, source)
        assert signature == "function myFunc(arg1: string, arg2: number): boolean"

    def test_should_extract_arrow_function_signature_with_params(self, typescript_parser):
        source = b"const myFunc = (x: number): string => x.toString();"

        params_node = Mock(spec=Node)
        params_node.start_byte = 15

        arrow_node = Mock(spec=Node)
        arrow_node.type = "=>"
        arrow_node.end_byte = 38

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.start_byte = 15
        node.child_by_field_name = Mock(return_value=params_node)
        node.children = [params_node, arrow_node]

        signature = typescript_parser._get_signature(node, source)
        assert signature == "(x: number): string =>"

    def test_should_extract_arrow_function_signature_with_parameter(self, typescript_parser):
        source = b"const myFunc = x => x * 2;"

        param_node = Mock(spec=Node)
        param_node.start_byte = 15

        arrow_node = Mock(spec=Node)
        arrow_node.type = "=>"
        arrow_node.end_byte = 19

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.start_byte = 15
        node.child_by_field_name = Mock(side_effect=lambda field: param_node if field == "parameter" else None)
        node.children = [param_node, arrow_node]

        signature = typescript_parser._get_signature(node, source)
        assert "x =>" in signature

    def test_should_extract_arrow_function_fallback_to_body(self, typescript_parser):
        source = b"const myFunc = (x) => { return x; };"

        body_node = Mock(spec=Node)
        body_node.start_byte = 22

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.start_byte = 15
        node.child_by_field_name = Mock(side_effect=lambda field: body_node if field == "body" else None)
        node.children = []

        signature = typescript_parser._get_signature(node, source)
        assert signature == "(x) =>"

    def test_should_extract_method_signature(self, typescript_parser):
        source = b"public myMethod(arg: string): void { }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 35

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = typescript_parser._get_signature(node, source)
        assert signature == "public myMethod(arg: string): void"

    def test_should_extract_method_signature_from_interface(self, typescript_parser):
        source = b"myMethod(arg: string): void;"

        node = Mock(spec=Node)
        node.type = "method_signature"
        node.start_byte = 0
        node.end_byte = 28
        node.child_by_field_name = Mock(return_value=None)

        signature = typescript_parser._get_signature(node, source)
        assert signature == "myMethod(arg: string): void;"

    def test_should_return_none_for_non_function_node(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"

        signature = typescript_parser._get_signature(node, b"class MyClass { }")
        assert signature is None

    def test_should_handle_generator_function(self, typescript_parser):
        source = b"function* generator() { yield 1; }"

        body_node = Mock(spec=Node)
        body_node.start_byte = 22

        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = typescript_parser._get_signature(node, source)
        assert signature == "function* generator()"


class TestIsAsync:
    """Test the _is_async helper method."""

    def test_should_return_true_for_async_function(self, typescript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [async_child, Mock(spec=Node, type="identifier")]

        assert typescript_parser._is_async(node)

    def test_should_return_false_for_sync_function(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [
            Mock(spec=Node, type="function"),
            Mock(spec=Node, type="identifier"),
        ]

        assert not typescript_parser._is_async(node)

    def test_should_return_true_for_async_arrow_function(self, typescript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.children = [async_child]

        assert typescript_parser._is_async(node)

    def test_should_return_true_for_async_method(self, typescript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [async_child, Mock(spec=Node, type="identifier")]

        assert typescript_parser._is_async(node)

    def test_should_return_false_for_non_function_node(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = []

        assert not typescript_parser._is_async(node)

    def test_should_return_true_for_async_generator(self, typescript_parser):
        async_child = Mock(spec=Node)
        async_child.type = "async"

        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.children = [async_child]

        assert typescript_parser._is_async(node)


class TestIsGenerator:
    """Test the _is_generator helper method."""

    def test_should_return_true_for_generator_function_declaration(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "generator_function_declaration"

        assert typescript_parser._is_generator(node)

    def test_should_return_true_for_function_with_star(self, typescript_parser):
        star_child = Mock(spec=Node)
        star_child.type = "*"

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [Mock(spec=Node, type="function"), star_child]

        assert typescript_parser._is_generator(node)

    def test_should_return_true_for_method_with_star(self, typescript_parser):
        star_child = Mock(spec=Node)
        star_child.type = "*"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [star_child, Mock(spec=Node, type="identifier")]

        assert typescript_parser._is_generator(node)

    def test_should_return_false_for_regular_function(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [Mock(spec=Node, type="identifier")]

        assert not typescript_parser._is_generator(node)

    def test_should_return_false_for_arrow_function(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.children = []

        assert not typescript_parser._is_generator(node)


class TestIsAbstract:
    """Test the _is_abstract helper method."""

    def test_should_return_true_for_abstract_class_declaration(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "abstract_class_declaration"
        node.children = []

        assert typescript_parser._is_abstract(node)

    def test_should_return_true_for_method_with_abstract_keyword(self, typescript_parser):
        abstract_child = Mock(spec=Node)
        abstract_child.type = "abstract"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [abstract_child, Mock(spec=Node, type="identifier")]

        assert typescript_parser._is_abstract(node)

    def test_should_return_false_for_regular_class(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = [Mock(spec=Node, type="identifier")]

        assert not typescript_parser._is_abstract(node)

    def test_should_return_false_for_regular_method(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [Mock(spec=Node, type="identifier")]

        assert not typescript_parser._is_abstract(node)


class TestGetDecorators:
    """Test the _get_decorators helper method."""

    def test_should_extract_single_decorator(self, typescript_parser):
        source = b"@Component\nclass MyComponent { }"

        decorator = Mock(spec=Node)
        decorator.type = "decorator"
        decorator.text = b"@Component"

        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = [decorator, Mock(spec=Node, type="identifier")]
        node.parent = None

        decorators = typescript_parser._get_decorators(node, source)
        assert decorators == ["@Component"]

    def test_should_extract_multiple_decorators(self, typescript_parser):
        source = b"@Injectable()\n@Component\nclass MyService { }"

        dec1 = Mock(spec=Node)
        dec1.type = "decorator"
        dec1.text = b"@Injectable()"

        dec2 = Mock(spec=Node)
        dec2.type = "decorator"
        dec2.text = b"@Component"

        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = [dec1, dec2, Mock(spec=Node, type="identifier")]
        node.parent = None

        decorators = typescript_parser._get_decorators(node, source)
        assert decorators == ["@Injectable()", "@Component"]

    def test_should_return_empty_list_when_no_decorators(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = [Mock(spec=Node, type="identifier")]
        node.parent = None

        decorators = typescript_parser._get_decorators(node, b"class MyClass { }")
        assert decorators == []

    def test_should_extract_decorators_from_parent_siblings(self, typescript_parser):
        source = b"@Decorator\nfunction myFunc() { }"

        decorator = Mock(spec=Node)
        decorator.type = "decorator"
        decorator.text = b"@Decorator"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"
        func_node.children = []

        parent = Mock(spec=Node)
        parent.children = [decorator, func_node]

        func_node.parent = parent

        decorators = typescript_parser._get_decorators(func_node, source)
        # Should find decorator in parent siblings before the node
        assert "@Decorator" in decorators

    def test_should_handle_decorator_without_text(self, typescript_parser):
        decorator = Mock(spec=Node)
        decorator.type = "decorator"
        decorator.text = None

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [decorator]
        node.parent = None

        decorators = typescript_parser._get_decorators(node, b"@Dec method() { }")
        assert decorators == []


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_return_extra_for_async_function(self, typescript_parser):
        source = b"async function fetch() { }"

        async_node = Mock(spec=Node)
        async_node.type = "async"

        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = [async_node]
        node.parent = None

        extra = typescript_parser._get_extra(node, source)
        assert extra["is_async"] == "true"
        assert extra["is_generator"] == "false"
        assert extra["is_arrow"] == "false"
        assert extra["is_abstract"] == "false"

    def test_should_return_extra_for_generator_function(self, typescript_parser):
        source = b"function* gen() { yield 1; }"

        node = Mock(spec=Node)
        node.type = "generator_function_declaration"
        node.children = []
        node.parent = None

        extra = typescript_parser._get_extra(node, source)
        assert extra["is_generator"] == "true"
        assert extra["is_async"] == "false"

    def test_should_return_extra_for_arrow_function(self, typescript_parser):
        source = b"const func = () => {}"

        node = Mock(spec=Node)
        node.type = "arrow_function"
        node.children = []
        node.parent = None

        extra = typescript_parser._get_extra(node, source)
        assert extra["is_arrow"] == "true"

    def test_should_return_extra_for_abstract_method(self, typescript_parser):
        source = b"abstract class A { abstract method(): void; }"

        abstract_node = Mock(spec=Node)
        abstract_node.type = "abstract"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [abstract_node]
        node.parent = None

        extra = typescript_parser._get_extra(node, source)
        assert extra["is_abstract"] == "true"

    def test_should_return_extra_with_decorators(self, typescript_parser):
        source = b"@dec1\n@dec2\nclass MyClass { }"

        dec1 = Mock(spec=Node)
        dec1.type = "decorator"
        dec1.text = b"@dec1"

        dec2 = Mock(spec=Node)
        dec2.type = "decorator"
        dec2.text = b"@dec2"

        node = Mock(spec=Node)
        node.type = "class_declaration"
        node.children = [dec1, dec2]
        node.parent = None

        extra = typescript_parser._get_extra(node, source)
        assert extra["decorators"] == "@dec1,@dec2"

    def test_should_return_extra_with_visibility(self, typescript_parser):
        source = b"private method() { }"

        visibility_node = Mock(spec=Node)
        visibility_node.type = "accessibility_modifier"
        visibility_node.text = b"private"

        node = Mock(spec=Node)
        node.type = "method_definition"
        node.children = [visibility_node]
        node.parent = None

        extra = typescript_parser._get_extra(node, source)
        assert extra["visibility"] == "private"

    def test_should_return_empty_strings_for_missing_attributes(self, typescript_parser):
        node = Mock(spec=Node)
        node.type = "function_declaration"
        node.children = []
        node.parent = None

        extra = typescript_parser._get_extra(node, b"function simple() { }")
        assert extra["decorators"] == ""
        assert extra["visibility"] == ""
        assert extra["is_async"] == "false"
        assert extra["is_generator"] == "false"
        assert extra["is_arrow"] == "false"
        assert extra["is_abstract"] == "false"


class TestGetDocumentation:
    """Test the _get_documentation helper method."""

    def test_should_extract_jsdoc_comment_before_function(self, typescript_parser):
        source = b"/** Function doc */\nfunction myFunc() { }"

        comment_node = Mock(spec=Node)
        comment_node.type = "comment"
        comment_node.text = b"/** Function doc */"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]

        func_node.parent = parent

        doc = typescript_parser._get_documentation(func_node, source)
        assert doc == "Function doc"

    def test_should_return_none_when_no_preceding_comment(self, typescript_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"

        parent = Mock(spec=Node)
        parent.children = [func_node]

        func_node.parent = parent

        doc = typescript_parser._get_documentation(func_node, b"function myFunc() { }")
        assert doc is None

    def test_should_return_none_when_node_is_first_child(self, typescript_parser):
        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"

        parent = Mock(spec=Node)
        parent.children = [func_node, Mock(spec=Node)]

        func_node.parent = parent

        doc = typescript_parser._get_documentation(func_node, b"function myFunc() { }")
        assert doc is None

    def test_should_return_none_when_no_parent(self, typescript_parser):
        node = Mock(spec=Node)
        node.parent = None

        doc = typescript_parser._get_documentation(node, b"function myFunc() { }")
        assert doc is None

    def test_should_return_none_when_prev_sibling_not_comment(self, typescript_parser):
        other_node = Mock(spec=Node)
        other_node.type = "identifier"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"

        parent = Mock(spec=Node)
        parent.children = [other_node, func_node]

        func_node.parent = parent

        doc = typescript_parser._get_documentation(func_node, b"x; function myFunc() { }")
        assert doc is None

    def test_should_handle_multiline_jsdoc(self, typescript_parser):
        source = b"/**\n * Multi\n * line\n */\nfunction f() { }"

        comment_node = Mock(spec=Node)
        comment_node.type = "comment"
        comment_node.text = b"/**\n * Multi\n * line\n */"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"

        parent = Mock(spec=Node)
        parent.children = [comment_node, func_node]

        func_node.parent = parent

        doc = typescript_parser._get_documentation(func_node, source)
        assert "Multi" in doc
        assert "line" in doc


class TestGetExportName:
    """Test the _get_export_name helper method."""

    def test_should_extract_name_from_exported_function(self, typescript_parser):
        source = b"export function myFunc() { }"

        name_node = Mock(spec=Node)
        name_node.text = b"myFunc"

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"
        func_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), func_node]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "myFunc"

    def test_should_extract_name_from_exported_class(self, typescript_parser):
        source = b"export class MyClass { }"

        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), class_node]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "MyClass"

    def test_should_extract_name_from_exported_interface(self, typescript_parser):
        source = b"export interface MyInterface { }"

        name_node = Mock(spec=Node)
        name_node.text = b"MyInterface"

        interface_node = Mock(spec=Node)
        interface_node.type = "interface_declaration"
        interface_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), interface_node]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "MyInterface"

    def test_should_extract_name_from_exported_type(self, typescript_parser):
        source = b"export type MyType = string;"

        name_node = Mock(spec=Node)
        name_node.text = b"MyType"

        type_node = Mock(spec=Node)
        type_node.type = "type_alias_declaration"
        type_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), type_node]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "MyType"

    def test_should_extract_identifier_name(self, typescript_parser):
        source = b"export { myVar };"

        identifier = Mock(spec=Node)
        identifier.type = "identifier"
        identifier.text = b"myVar"

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), identifier]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "myVar"

    def test_should_return_default_when_no_name_found(self, typescript_parser):
        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export")]

        name = typescript_parser._get_export_name(export_node, b"export default something;")
        assert name == "default"

    def test_should_extract_name_from_lexical_declaration(self, typescript_parser):
        source = b"export const myConst = 123;"

        name_node = Mock(spec=Node)
        name_node.text = b"myConst"

        var_decl = Mock(spec=Node)
        var_decl.type = "variable_declarator"
        var_decl.child_by_field_name = Mock(return_value=name_node)

        lex_decl = Mock(spec=Node)
        lex_decl.type = "lexical_declaration"
        lex_decl.children = [Mock(spec=Node, type="const"), var_decl]

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), lex_decl]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "myConst"

    def test_should_extract_multiple_names_from_export_clause(self, typescript_parser):
        source = b"export { a, b, c };"

        name1 = Mock(spec=Node)
        name1.text = b"a"

        name2 = Mock(spec=Node)
        name2.text = b"b"

        name3 = Mock(spec=Node)
        name3.text = b"c"

        spec1 = Mock(spec=Node)
        spec1.type = "export_specifier"
        spec1.child_by_field_name = Mock(return_value=name1)

        spec2 = Mock(spec=Node)
        spec2.type = "export_specifier"
        spec2.child_by_field_name = Mock(return_value=name2)

        spec3 = Mock(spec=Node)
        spec3.type = "export_specifier"
        spec3.child_by_field_name = Mock(return_value=name3)

        clause = Mock(spec=Node)
        clause.type = "export_clause"
        clause.children = [spec1, spec2, spec3]

        export_node = Mock(spec=Node)
        export_node.children = [Mock(spec=Node, type="export"), clause]

        name = typescript_parser._get_export_name(export_node, source)
        assert name == "a, b, c"


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, typescript_parser):
        match = {"name": [Mock(spec=Node)]}
        result = typescript_parser.process_match(match, b"")
        assert result is None

    def test_should_skip_lexical_declaration_inside_export(self, typescript_parser):
        export_node = Mock(spec=Node)
        export_node.type = "export_statement"

        lex_node = Mock(spec=Node)
        lex_node.type = "lexical_declaration"
        lex_node.parent = export_node

        match = {"def": [lex_node]}
        result = typescript_parser.process_match(match, b"export const x = 1;")
        assert result is None

    def test_should_skip_non_const_lexical_declaration(self, typescript_parser):
        lex_node = Mock(spec=Node)
        lex_node.type = "lexical_declaration"
        lex_node.parent = None
        lex_node.children = [Mock(spec=Node, type="let")]

        name_node = Mock(spec=Node)
        name_node.text = b"myVar"

        match = {"def": [lex_node], "name": [name_node]}
        result = typescript_parser.process_match(match, b"let myVar = 1;")
        assert result is None

    def test_should_skip_lowercase_const_without_arrow_function(self, typescript_parser):
        const_node = Mock(spec=Node)
        const_node.type = "const"

        lex_node = Mock(spec=Node)
        lex_node.type = "lexical_declaration"
        lex_node.parent = None
        lex_node.children = [const_node]

        name_node = Mock(spec=Node)
        name_node.text = b"myVar"

        match = {"def": [lex_node], "name": [name_node]}
        result = typescript_parser.process_match(match, b"const myVar = 1;")
        assert result is None

    def test_should_process_uppercase_constant(self, typescript_parser):
        const_node = Mock(spec=Node)
        const_node.type = "const"

        lex_node = Mock(spec=Node)
        lex_node.type = "lexical_declaration"
        lex_node.parent = None
        lex_node.start_byte = 0
        lex_node.end_byte = 19
        lex_node.start_point = (0, 0)
        lex_node.end_point = (0, 19)
        lex_node.children = [const_node]
        lex_node.child_by_field_name = Mock(return_value=None)

        name_node = Mock(spec=Node)
        name_node.text = b"MY_CONSTANT"

        match = {"def": [lex_node], "name": [name_node]}
        source = b"const MY_CONSTANT = 1;"

        result = typescript_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert "MY_CONSTANT" in content
        assert node_info["node_name"] == "MY_CONSTANT"
        assert node_info["node_type"] == "constant"

    def test_should_process_arrow_function(self, typescript_parser):
        arrow_node = Mock(spec=Node)
        arrow_node.type = "arrow_function"
        arrow_node.parent = None
        arrow_node.children = []
        arrow_node.start_byte = 15
        arrow_node.child_by_field_name = Mock(return_value=None)

        const_node = Mock(spec=Node)
        const_node.type = "const"

        lex_node = Mock(spec=Node)
        lex_node.type = "lexical_declaration"
        lex_node.parent = None
        lex_node.start_byte = 0
        lex_node.end_byte = 25
        lex_node.start_point = (0, 0)
        lex_node.end_point = (0, 25)
        lex_node.children = [const_node]
        lex_node.child_by_field_name = Mock(return_value=None)

        name_node = Mock(spec=Node)
        name_node.text = b"myFunc"

        match = {"def": [lex_node], "name": [name_node], "arrow_func": [arrow_node]}
        source = b"const myFunc = () => {};"

        result = typescript_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "myFunc"
        assert node_info["extra"]["is_arrow"] == "true"

    def test_should_process_simple_function(self, typescript_parser):
        body_node = Mock(spec=Node)
        body_node.children = []
        body_node.start_byte = 20

        func_node = Mock(spec=Node)
        func_node.type = "function_declaration"
        func_node.parent = None
        func_node.start_byte = 0
        func_node.end_byte = 25
        func_node.start_point = (0, 0)
        func_node.end_point = (0, 25)
        func_node.children = []
        func_node.child_by_field_name = Mock(return_value=body_node)

        name_node = Mock(spec=Node)
        name_node.text = b"myFunc"

        match = {"def": [func_node], "name": [name_node]}
        source = b"function myFunc() { }"

        result = typescript_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert "myFunc" in content
        assert node_info["node_name"] == "myFunc"
        assert node_info["node_type"] == "function"

    def test_should_process_import_statement_with_source(self, typescript_parser):
        source_node = Mock(spec=Node)
        source_node.text = b"'./module'"

        import_node = Mock(spec=Node)
        import_node.type = "import_statement"
        import_node.parent = None
        import_node.start_byte = 0
        import_node.end_byte = 30
        import_node.start_point = (0, 0)
        import_node.end_point = (0, 30)
        import_node.children = []
        import_node.child_by_field_name = Mock(return_value=None)

        match = {"def": [import_node], "source": [source_node]}
        source = b"import { x } from './module';"

        result = typescript_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "./module"
        assert node_info["node_type"] == "import"

    def test_should_process_export_statement(self, typescript_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_declaration"
        class_node.child_by_field_name = Mock(return_value=name_node)

        export_node = Mock(spec=Node)
        export_node.type = "export_statement"
        export_node.parent = None
        export_node.start_byte = 0
        export_node.end_byte = 30
        export_node.start_point = (0, 0)
        export_node.end_point = (1, 0)
        export_node.children = [Mock(spec=Node, type="export"), class_node]
        export_node.child_by_field_name = Mock(return_value=None)

        match = {"def": [export_node]}
        source = b"export class MyClass { }"

        result = typescript_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "MyClass"
        assert node_info["node_type"] == "export"


class TestTypeScriptParserInitialization:
    """Test TypeScriptParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = TypeScriptParser()
        assert parser.language == "typescript"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, typescript_parser):
        query = typescript_parser.query_str
        assert "function_declaration" in query
        assert "class_declaration" in query
        assert "interface_declaration" in query
        assert "type_alias_declaration" in query
        assert "enum_declaration" in query
        assert "method_definition" in query
        assert "arrow_function" in query
        assert "import_statement" in query
        assert "export_statement" in query


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real TypeScript code."""

    def test_should_parse_simple_function(self, typescript_parser):
        content = """function helloWorld(): void {
    console.log("Hello, World!");
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        content_str, node_metadata = results[0]
        assert "helloWorld" in content_str
        assert node_metadata.node_name == "helloWorld"
        assert node_metadata.node_type == "function"
        assert node_metadata.language == "typescript"

    def test_should_parse_function_with_jsdoc(self, typescript_parser):
        content = """/**
 * This function has documentation.
 */
function documented(): number {
    return 42;
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.documentation == "This function has documentation."

    def test_should_parse_class_definition(self, typescript_parser):
        content = """class MyClass {
    constructor() {}
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find class + constructor method
        assert len(results) >= 1
        class_result = [r for r in results if r[1].node_name == "MyClass"][0]
        assert class_result[1].node_type == "class"

    def test_should_parse_class_with_methods(self, typescript_parser):
        content = """class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
    
    subtract(a: number, b: number): number {
        return a - b;
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find class + 2 methods
        assert len(results) == 3

        class_result = [r for r in results if r[1].node_name == "Calculator"][0]
        assert class_result[1].node_type == "class"

        add_result = [r for r in results if r[1].node_name == "add"][0]
        assert add_result[1].node_type == "method"
        assert add_result[1].parent_scope == "Calculator"

        subtract_result = [r for r in results if r[1].node_name == "subtract"][0]
        assert subtract_result[1].node_type == "method"
        assert subtract_result[1].parent_scope == "Calculator"

    def test_should_parse_interface_definition(self, typescript_parser):
        content = """interface User {
    name: string;
    age: number;
    greet(): void;
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find interface + method signature
        assert len(results) >= 1
        interface_result = [r for r in results if r[1].node_name == "User"][0]
        assert interface_result[1].node_type == "interface"

    def test_should_parse_type_alias(self, typescript_parser):
        content = """type Status = 'pending' | 'approved' | 'rejected';"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Status"
        assert node_metadata.node_type == "type_alias"

    def test_should_parse_enum_definition(self, typescript_parser):
        content = """enum Color {
    Red,
    Green,
    Blue
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Color"
        assert node_metadata.node_type == "enum"

    def test_should_parse_arrow_function(self, typescript_parser):
        content = """const multiply = (x: number, y: number): number => x * y;"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "multiply"
        assert node_metadata.extra["is_arrow"] == "true"

    def test_should_parse_async_function(self, typescript_parser):
        content = """async function fetchData(): Promise<string> {
    return await fetch('/api');
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "fetchData"
        assert node_metadata.extra["is_async"] == "true"

    def test_should_parse_generator_function(self, typescript_parser):
        content = """function* numberGenerator(): Generator<number> {
    yield 1;
    yield 2;
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "numberGenerator"
        assert node_metadata.extra["is_generator"] == "true"

    def test_should_parse_abstract_class(self, typescript_parser):
        content = """abstract class Animal {
    abstract makeSound(): void;
    
    move(): void {
        console.log("Moving...");
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find abstract class + 2 methods
        assert len(results) >= 1
        class_result = [r for r in results if r[1].node_name == "Animal"][0]
        assert class_result[1].node_type == "class"
        assert class_result[1].extra["is_abstract"] == "true"

    def test_should_parse_class_with_decorators(self, typescript_parser):
        content = """@Component({
    selector: 'app-root'
})
class AppComponent {
    title = 'my-app';
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) >= 1
        class_result = [r for r in results if r[1].node_name == "AppComponent"][0]
        assert "@Component" in class_result[1].extra["decorators"]

    def test_should_parse_class_with_visibility_modifiers(self, typescript_parser):
        content = """class Person {
    private name: string;
    
    public getName(): string {
        return this.name;
    }
    
    protected setName(name: string): void {
        this.name = name;
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        methods = [r for r in results if r[1].node_type == "method"]

        get_name = [m for m in methods if m[1].node_name == "getName"][0]
        assert get_name[1].extra["visibility"] == "public"

        set_name = [m for m in methods if m[1].node_name == "setName"][0]
        assert set_name[1].extra["visibility"] == "protected"

    def test_should_parse_module_constants(self, typescript_parser):
        content = """const API_URL = "https://api.example.com";
const MAX_RETRIES = 3;
const baseUrl = "http://example.com";"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should only find uppercase constants
        constant_names = {r[1].node_name for r in results}
        assert "API_URL" in constant_names
        assert "MAX_RETRIES" in constant_names
        assert "baseUrl" not in constant_names

    def test_should_parse_import_statements(self, typescript_parser):
        content = """import { Component } from '@angular/core';
import * as fs from 'fs';"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 2
        import_names = {r[1].node_name for r in results}
        assert "@angular/core" in import_names
        assert "fs" in import_names

    def test_should_parse_export_statements(self, typescript_parser):
        content = """export function helper() {}
export class Utils {}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find: export statements (2) + the function and class themselves (2) = 4
        assert len(results) == 4
        export_results = [r for r in results if r[1].node_type == "export"]
        assert len(export_results) == 2

    def test_should_handle_complex_code_with_multiple_elements(self, typescript_parser):
        content = """import { Injectable } from '@angular/core';

const API_KEY = 'secret';

/**
 * Service for data processing.
 */
@Injectable()
export class DataService {
    private data: string[] = [];
    
    /**
     * Add item to data.
     */
    public add(item: string): void {
        this.data.push(item);
    }
    
    async fetch(): Promise<void> {
        // Implementation
    }
}

export function helper(): string {
    return 'help';
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find: import, constant, export (class), class, methods, export (function)
        assert len(results) >= 5

        # Verify constant
        constants = [r for r in results if r[1].node_type == "constant"]
        assert len(constants) == 1
        assert constants[0][1].node_name == "API_KEY"

        # Verify class
        classes = [r for r in results if r[1].node_type == "class"]
        assert len(classes) == 1
        assert classes[0][1].node_name == "DataService"

        # Verify class has decorator
        class_result = classes[0]
        assert "@Injectable" in class_result[1].extra["decorators"]

        # Verify methods
        methods = [r for r in results if r[1].node_type == "method"]
        assert len(methods) >= 2
        method_names = {m[1].node_name for m in methods}
        assert "add" in method_names
        assert "fetch" in method_names

        # Verify async method
        fetch_method = [m for m in methods if m[1].node_name == "fetch"][0]
        assert fetch_method[1].extra["is_async"] == "true"

        # Verify visibility
        add_method = [m for m in methods if m[1].node_name == "add"][0]
        assert add_method[1].extra["visibility"] == "public"

    def test_should_handle_empty_file(self, typescript_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))
        assert len(results) == 0

    def test_should_handle_file_with_only_comments(self, typescript_parser):
        content = """// This is a comment
/* Another comment */"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="comments.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))
        assert len(results) == 0

    def test_should_include_metadata_fields_from_document(self, typescript_parser):
        content = """function testFunc(): void {}"""
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/module.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/module.ts"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_nested_classes(self, typescript_parser):
        content = """class Outer {
    static Inner = class {
        method() {}
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find both classes
        assert len(results) >= 1

    def test_should_handle_method_overloads(self, typescript_parser):
        content = """class Calculator {
    add(a: number, b: number): number;
    add(a: string, b: string): string;
    add(a: any, b: any): any {
        return a + b;
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find class and method(s)
        assert len(results) >= 1

    def test_should_parse_generic_class(self, typescript_parser):
        content = """class Container<T> {
    private value: T;
    
    get(): T {
        return this.value;
    }
}"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".ts",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.ts", content=content, metadata=metadata)

        results = list(typescript_parser.parse(document))

        # Should find class and method
        assert len(results) >= 2
        class_result = [r for r in results if r[1].node_name == "Container"][0]
        assert class_result[1].node_type == "class"
