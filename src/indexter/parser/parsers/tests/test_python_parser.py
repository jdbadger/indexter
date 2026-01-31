from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parser.parsers.python import PythonParser
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def python_parser():
    """Create a PythonParser instance for testing."""
    return PythonParser()


@pytest.fixture
def sample_python_document():
    """Create a sample Python Document for testing."""
    content = """
def simple_function():
    pass

class SimpleClass:
    pass
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        ext=".py",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.py",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestIsConstant:
    """Test the _is_constant helper method."""

    def test_should_return_true_for_all_uppercase(self, python_parser):
        assert python_parser._is_constant("CONSTANT")

    def test_should_return_true_for_uppercase_with_underscores(self, python_parser):
        assert python_parser._is_constant("MY_CONSTANT")
        assert python_parser._is_constant("API_KEY")
        assert python_parser._is_constant("MAX_VALUE_123")

    def test_should_return_false_for_lowercase(self, python_parser):
        assert not python_parser._is_constant("variable")
        assert not python_parser._is_constant("my_var")

    def test_should_return_false_for_mixed_case(self, python_parser):
        assert not python_parser._is_constant("MyClass")
        assert not python_parser._is_constant("myVariable")
        assert not python_parser._is_constant("My_Constant")

    def test_should_return_false_for_single_lowercase_letter(self, python_parser):
        assert not python_parser._is_constant("a")

    def test_should_return_true_for_single_uppercase_letter(self, python_parser):
        assert python_parser._is_constant("A")


class TestStripDocstring:
    """Test the _strip_docstring helper method."""

    def test_should_strip_triple_double_quotes(self, python_parser):
        docstring = '"""This is a docstring"""'
        assert python_parser._strip_docstring(docstring) == "This is a docstring"

    def test_should_strip_triple_single_quotes(self, python_parser):
        docstring = "'''This is a docstring'''"
        assert python_parser._strip_docstring(docstring) == "This is a docstring"

    def test_should_strip_single_double_quotes(self, python_parser):
        docstring = '"Simple string"'
        assert python_parser._strip_docstring(docstring) == "Simple string"

    def test_should_strip_single_quotes(self, python_parser):
        docstring = "'Simple string'"
        assert python_parser._strip_docstring(docstring) == "Simple string"

    def test_should_handle_multiline_docstrings(self, python_parser):
        docstring = '"""This is\na multiline\ndocstring"""'
        assert python_parser._strip_docstring(docstring) == "This is\na multiline\ndocstring"

    def test_should_strip_leading_and_trailing_whitespace(self, python_parser):
        docstring = '"""  Docstring with whitespace  """'
        assert python_parser._strip_docstring(docstring) == "Docstring with whitespace"

    def test_should_handle_empty_docstring(self, python_parser):
        docstring = '""""""'
        assert python_parser._strip_docstring(docstring) == ""

    def test_should_not_strip_mismatched_quotes(self, python_parser):
        docstring = '"""This is not properly closed"'
        # Should return stripped version but original text content
        result = python_parser._strip_docstring(docstring)
        assert '"' not in result or result.startswith('"')


class TestIsAsync:
    """Test the _is_async helper method."""

    def test_should_return_true_for_async_function(self, python_parser):
        # Create a mock node structure for async function
        node = Mock(spec=Node)
        node.type = "function_definition"
        async_child = Mock(spec=Node)
        async_child.type = "async"
        regular_child = Mock(spec=Node)
        regular_child.type = "identifier"
        node.children = [async_child, regular_child]

        assert python_parser._is_async(node)

    def test_should_return_false_for_sync_function(self, python_parser):
        node = Mock(spec=Node)
        node.type = "function_definition"
        node.children = [
            Mock(spec=Node, type="identifier"),
            Mock(spec=Node, type="parameters"),
        ]

        assert not python_parser._is_async(node)

    def test_should_handle_decorated_async_function(self, python_parser):
        # Create mock decorated definition with inner async function
        inner_func = Mock(spec=Node)
        inner_func.type = "function_definition"
        async_child = Mock(spec=Node, type="async")
        inner_func.children = [async_child, Mock(spec=Node, type="identifier")]

        decorated_node = Mock(spec=Node)
        decorated_node.type = "decorated_definition"
        decorated_node.children = [
            Mock(spec=Node, type="decorator"),
            inner_func,
        ]

        assert python_parser._is_async(decorated_node)

    def test_should_return_false_for_decorated_sync_function(self, python_parser):
        inner_func = Mock(spec=Node)
        inner_func.type = "function_definition"
        inner_func.children = [Mock(spec=Node, type="identifier")]

        decorated_node = Mock(spec=Node)
        decorated_node.type = "decorated_definition"
        decorated_node.children = [Mock(spec=Node, type="decorator"), inner_func]

        assert not python_parser._is_async(decorated_node)

    def test_should_return_false_for_non_function_node(self, python_parser):
        node = Mock(spec=Node)
        node.type = "class_definition"
        node.children = []

        assert not python_parser._is_async(node)


class TestGetDecorators:
    """Test the _get_decorators helper method."""

    def test_should_extract_decorators_from_decorated_definition(self, python_parser):
        source = b"@decorator1\n@decorator2\ndef func():\n    pass"

        decorator1 = Mock(spec=Node)
        decorator1.type = "decorator"
        decorator1.start_byte = 0
        decorator1.end_byte = 11

        decorator2 = Mock(spec=Node)
        decorator2.type = "decorator"
        decorator2.start_byte = 12
        decorator2.end_byte = 23

        other_child = Mock(spec=Node)
        other_child.type = "function_definition"

        node = Mock(spec=Node)
        node.type = "decorated_definition"
        node.children = [decorator1, decorator2, other_child]

        decorators = python_parser._get_decorators(node, source)
        assert decorators == ["@decorator1", "@decorator2"]

    def test_should_return_empty_list_for_non_decorated_node(self, python_parser):
        node = Mock(spec=Node)
        node.type = "function_definition"
        node.children = []

        decorators = python_parser._get_decorators(node, b"def func(): pass")
        assert decorators == []

    def test_should_handle_decorated_definition_with_no_decorators(self, python_parser):
        node = Mock(spec=Node)
        node.type = "decorated_definition"
        node.children = [Mock(spec=Node, type="function_definition")]

        decorators = python_parser._get_decorators(node, b"def func(): pass")
        assert decorators == []


class TestGetContent:
    """Test the _get_content helper method."""

    def test_should_extract_node_content(self, python_parser):
        source = b"def hello():\n    print('hello')"
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 31

        content = python_parser._get_content(node, source)
        assert content == "def hello():\n    print('hello')"

    def test_should_extract_partial_content(self, python_parser):
        source = b"# Comment\ndef hello():\n    pass"
        node = Mock(spec=Node)
        node.start_byte = 10
        node.end_byte = 34

        content = python_parser._get_content(node, source)
        assert content == "def hello():\n    pass"


class TestGetNodeType:
    """Test the _get_node_type helper method."""

    def test_should_return_function_for_module_level_function(self, python_parser):
        node = Mock(spec=Node)
        node.parent = None

        node_type = python_parser._get_node_type("function_definition", node)
        assert node_type == "function"

    def test_should_return_method_for_class_function(self, python_parser):
        class_node = Mock(spec=Node)
        class_node.type = "class_definition"
        class_node.child_by_field_name = Mock(return_value=Mock(text=b"MyClass"))
        class_node.parent = None

        node = Mock(spec=Node)
        node.parent = class_node

        node_type = python_parser._get_node_type("function_definition", node)
        assert node_type == "method"

    def test_should_return_class_for_class_definition(self, python_parser):
        node = Mock(spec=Node)
        node_type = python_parser._get_node_type("class_definition", node)
        assert node_type == "class"

    def test_should_return_constant_for_assignment(self, python_parser):
        node = Mock(spec=Node)
        node_type = python_parser._get_node_type("assignment", node)
        assert node_type == "constant"

    def test_should_return_import_for_import_statement(self, python_parser):
        node = Mock(spec=Node)
        node_type = python_parser._get_node_type("import_statement", node)
        assert node_type == "import"

    def test_should_return_import_for_import_from_statement(self, python_parser):
        node = Mock(spec=Node)
        node_type = python_parser._get_node_type("import_from_statement", node)
        assert node_type == "import"

    def test_should_return_original_type_for_unknown(self, python_parser):
        node = Mock(spec=Node)
        node_type = python_parser._get_node_type("unknown_type", node)
        assert node_type == "unknown_type"


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_class_name_for_method(self, python_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MyClass"

        class_node = Mock(spec=Node)
        class_node.type = "class_definition"
        class_node.child_by_field_name = Mock(return_value=name_node)
        class_node.parent = None

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = python_parser._get_parent_scope(method_node)
        assert parent_scope == "MyClass"

    def test_should_return_none_for_module_level_function(self, python_parser):
        node = Mock(spec=Node)
        node.parent = None

        parent_scope = python_parser._get_parent_scope(node)
        assert parent_scope is None

    def test_should_traverse_multiple_levels(self, python_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"OuterClass"

        outer_class = Mock(spec=Node)
        outer_class.type = "class_definition"
        outer_class.child_by_field_name = Mock(return_value=name_node)
        outer_class.parent = None

        block_node = Mock(spec=Node)
        block_node.type = "block"
        block_node.parent = outer_class

        inner_method = Mock(spec=Node)
        inner_method.parent = block_node

        parent_scope = python_parser._get_parent_scope(inner_method)
        assert parent_scope == "OuterClass"

    def test_should_handle_node_with_no_name(self, python_parser):
        class_node = Mock(spec=Node)
        class_node.type = "class_definition"
        class_node.child_by_field_name = Mock(return_value=None)
        class_node.parent = None

        method_node = Mock(spec=Node)
        method_node.parent = class_node

        parent_scope = python_parser._get_parent_scope(method_node)
        assert parent_scope is None


class TestGetSignature:
    """Test the _get_signature helper method."""

    def test_should_extract_function_signature(self, python_parser):
        source = b"def my_func(arg1, arg2):\n    pass"

        body_node = Mock(spec=Node)
        body_node.start_byte = 24

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.start_byte = 0
        node.end_byte = 33
        node.child_by_field_name = Mock(return_value=body_node)

        signature = python_parser._get_signature(node, source)
        assert signature == "def my_func(arg1, arg2)"

    def test_should_handle_signature_with_return_type(self, python_parser):
        source = b"def my_func(arg: int) -> str:\n    pass"

        body_node = Mock(spec=Node)
        body_node.start_byte = 29

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.start_byte = 0
        node.child_by_field_name = Mock(return_value=body_node)

        signature = python_parser._get_signature(node, source)
        assert signature == "def my_func(arg: int) -> str"

    def test_should_return_none_for_non_function(self, python_parser):
        node = Mock(spec=Node)
        node.type = "class_definition"

        signature = python_parser._get_signature(node, b"class MyClass: pass")
        assert signature is None

    def test_should_handle_function_without_body(self, python_parser):
        source = b"def my_func()"

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.start_byte = 0
        node.end_byte = 13
        node.child_by_field_name = Mock(return_value=None)

        signature = python_parser._get_signature(node, source)
        assert signature == "def my_func()"


class TestGetDocumentation:
    """Test the _get_documentation helper method."""

    def test_should_extract_function_docstring(self, python_parser):
        source = b'def func():\n    """This is a docstring"""\n    pass'

        string_node = Mock(spec=Node)
        string_node.type = "string"
        string_node.text = b'"""This is a docstring"""'

        expr_stmt = Mock(spec=Node)
        expr_stmt.type = "expression_statement"
        expr_stmt.children = [string_node]

        body_node = Mock(spec=Node)
        body_node.children = [expr_stmt]

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.child_by_field_name = Mock(return_value=body_node)

        doc = python_parser._get_documentation(node, source)
        assert doc == "This is a docstring"

    def test_should_extract_class_docstring(self, python_parser):
        source = b'class MyClass:\n    """Class docstring"""\n    pass'

        string_node = Mock(spec=Node)
        string_node.type = "string"
        string_node.text = b'"""Class docstring"""'

        expr_stmt = Mock(spec=Node)
        expr_stmt.type = "expression_statement"
        expr_stmt.children = [string_node]

        body_node = Mock(spec=Node)
        body_node.children = [expr_stmt]

        node = Mock(spec=Node)
        node.type = "class_definition"
        node.child_by_field_name = Mock(return_value=body_node)

        doc = python_parser._get_documentation(node, source)
        assert doc == "Class docstring"

    def test_should_handle_direct_string_literal(self, python_parser):
        source = b'def func():\n    """Docstring"""'

        string_node = Mock(spec=Node)
        string_node.type = "string"
        string_node.text = b'"""Docstring"""'

        body_node = Mock(spec=Node)
        body_node.children = [string_node]

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.child_by_field_name = Mock(return_value=body_node)

        doc = python_parser._get_documentation(node, source)
        assert doc == "Docstring"

    def test_should_return_none_for_function_without_docstring(self, python_parser):
        body_node = Mock(spec=Node)
        body_node.children = [Mock(spec=Node, type="pass_statement")]

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.child_by_field_name = Mock(return_value=body_node)

        doc = python_parser._get_documentation(node, b"def func():\n    pass")
        assert doc is None

    def test_should_return_none_for_non_function_or_class(self, python_parser):
        node = Mock(spec=Node)
        node.type = "assignment"

        doc = python_parser._get_documentation(node, b"x = 1")
        assert doc is None

    def test_should_return_none_for_empty_body(self, python_parser):
        body_node = Mock(spec=Node)
        body_node.children = []

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.child_by_field_name = Mock(return_value=body_node)

        doc = python_parser._get_documentation(node, b"def func(): ...")
        assert doc is None

    def test_should_return_none_when_no_body(self, python_parser):
        node = Mock(spec=Node)
        node.type = "function_definition"
        node.child_by_field_name = Mock(return_value=None)

        doc = python_parser._get_documentation(node, b"def func(): ...")
        assert doc is None


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_return_extra_for_decorated_async_function(self, python_parser):
        source = b"@decorator\nasync def func(): pass"

        decorator_node = Mock(spec=Node)
        decorator_node.type = "decorator"
        decorator_node.start_byte = 0
        decorator_node.end_byte = 10

        async_node = Mock(spec=Node, type="async")
        func_node = Mock(spec=Node, type="function_definition")
        func_node.children = [async_node]

        node = Mock(spec=Node)
        node.type = "decorated_definition"
        node.children = [decorator_node, func_node]

        extra = python_parser._get_extra(node, source)
        assert extra["decorators"] == "@decorator"
        assert extra["is_async"] == "true"

    def test_should_return_extra_for_sync_function_without_decorators(self, python_parser):
        node = Mock(spec=Node)
        node.type = "function_definition"
        node.children = []

        extra = python_parser._get_extra(node, b"def func(): pass")
        assert extra["decorators"] == ""
        assert extra["is_async"] == "false"

    def test_should_handle_multiple_decorators(self, python_parser):
        source = b"@dec1\n@dec2\n@dec3\ndef func(): pass"

        dec1 = Mock(spec=Node, type="decorator", start_byte=0, end_byte=5)
        dec2 = Mock(spec=Node, type="decorator", start_byte=6, end_byte=11)
        dec3 = Mock(spec=Node, type="decorator", start_byte=12, end_byte=17)
        func = Mock(spec=Node, type="function_definition")
        func.children = []

        node = Mock(spec=Node)
        node.type = "decorated_definition"
        node.children = [dec1, dec2, dec3, func]

        extra = python_parser._get_extra(node, source)
        assert extra["decorators"] == "@dec1,@dec2,@dec3"


# Unit tests for process_match


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_def_nodes(self, python_parser):
        match = {"name": [Mock(spec=Node)]}
        result = python_parser.process_match(match, b"")
        assert result is None

    def test_should_return_none_for_function_inside_decorated_definition(self, python_parser):
        parent = Mock(spec=Node)
        parent.type = "decorated_definition"

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.parent = parent

        match = {"def": [node]}
        result = python_parser.process_match(match, b"def func(): pass")
        assert result is None

    def test_should_return_none_for_class_inside_decorated_definition(self, python_parser):
        parent = Mock(spec=Node)
        parent.type = "decorated_definition"

        node = Mock(spec=Node)
        node.type = "class_definition"
        node.parent = parent

        match = {"def": [node]}
        result = python_parser.process_match(match, b"class MyClass: pass")
        assert result is None

    def test_should_return_none_when_no_name_or_module(self, python_parser):
        node = Mock(spec=Node)
        node.type = "function_definition"
        node.parent = None

        match = {"def": [node]}
        result = python_parser.process_match(match, b"def func(): pass")
        assert result is None

    def test_should_return_none_for_non_constant_assignment(self, python_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"my_var"

        node = Mock(spec=Node)
        node.type = "assignment"
        node.parent = None

        match = {"def": [node], "name": [name_node]}
        result = python_parser.process_match(match, b"my_var = 123")
        assert result is None

    def test_should_process_constant_assignment(self, python_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"MY_CONSTANT"

        node = Mock(spec=Node)
        node.type = "assignment"
        node.parent = None
        node.start_byte = 0
        node.end_byte = 17
        node.start_point = (0, 0)
        node.end_point = (0, 17)
        node.child_by_field_name = Mock(return_value=None)

        match = {"def": [node], "name": [name_node]}
        source = b"MY_CONSTANT = 123"

        result = python_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "MY_CONSTANT = 123"
        assert node_info["node_name"] == "MY_CONSTANT"
        assert node_info["node_type"] == "constant"

    def test_should_process_simple_function(self, python_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"my_func"

        body_node = Mock(spec=Node)
        body_node.children = []
        body_node.start_byte = 15

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.parent = None
        node.start_byte = 0
        node.end_byte = 24
        node.start_point = (0, 0)
        node.end_point = (1, 8)
        node.child_by_field_name = Mock(return_value=body_node)
        node.children = []

        match = {"def": [node], "name": [name_node]}
        source = b"def my_func():\n    pass"

        result = python_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert "my_func" in content
        assert node_info["node_name"] == "my_func"
        assert node_info["node_type"] == "function"
        assert node_info["parent_scope"] is None

    def test_should_process_import_from_statement_with_module(self, python_parser):
        module_node = Mock(spec=Node)
        module_node.text = b"os.path"

        node = Mock(spec=Node)
        node.type = "import_from_statement"
        node.parent = None
        node.start_byte = 0
        node.end_byte = 23
        node.start_point = (0, 0)
        node.end_point = (0, 23)
        node.child_by_field_name = Mock(return_value=None)

        match = {"def": [node], "module": [module_node]}
        source = b"from os.path import join"

        result = python_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "os.path"
        assert node_info["node_type"] == "import"

    def test_should_use_inner_node_for_decorated_definition(self, python_parser):
        name_node = Mock(spec=Node)
        name_node.text = b"my_func"

        body_node = Mock(spec=Node)
        body_node.children = []
        body_node.start_byte = 26

        inner_node = Mock(spec=Node)
        inner_node.type = "function_definition"
        inner_node.parent = None
        inner_node.start_byte = 11
        inner_node.child_by_field_name = Mock(return_value=body_node)
        inner_node.children = []

        decorator_node = Mock(spec=Node)
        decorator_node.type = "decorator"
        decorator_node.start_byte = 0
        decorator_node.end_byte = 11

        outer_node = Mock(spec=Node)
        outer_node.type = "decorated_definition"
        outer_node.parent = None
        outer_node.start_byte = 0
        outer_node.end_byte = 36
        outer_node.start_point = (0, 0)
        outer_node.end_point = (2, 8)
        outer_node.children = [decorator_node, inner_node]

        match = {"def": [outer_node], "name": [name_node], "inner": [inner_node]}
        source = b"@decorator\ndef my_func():\n    pass"

        result = python_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert node_info["node_name"] == "my_func"
        assert "@decorator" in node_info["extra"]["decorators"]


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real Python code."""

    def test_should_parse_simple_function(self, python_parser):
        content = """def hello_world():
    print("Hello, World!")
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        content_str, node_metadata = results[0]
        assert "hello_world" in content_str
        assert node_metadata.node_name == "hello_world"
        assert node_metadata.node_type == "function"
        assert node_metadata.language == "python"
        assert node_metadata.start_line == 1
        assert node_metadata.end_line == 2

    def test_should_parse_function_with_docstring(self, python_parser):
        content = '''def documented_func():
    """This function has documentation."""
    return 42
'''
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.documentation == "This function has documentation."

    def test_should_parse_class_definition(self, python_parser):
        content = """class MyClass:
    pass
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "MyClass"
        assert node_metadata.node_type == "class"

    def test_should_parse_class_with_methods(self, python_parser):
        content = """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

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

    def test_should_parse_async_function(self, python_parser):
        content = """async def fetch_data():
    return await some_api_call()
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "fetch_data"
        assert node_metadata.extra["is_async"] == "true"

    def test_should_parse_decorated_function(self, python_parser):
        content = """@app.route('/hello')
@login_required
def hello():
    return "Hello"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "hello"
        assert "@app.route('/hello')" in node_metadata.extra["decorators"]
        assert "@login_required" in node_metadata.extra["decorators"]

    def test_should_parse_module_constants(self, python_parser):
        content = """API_KEY = "secret"
MAX_RETRIES = 3
base_url = "http://example.com"
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        # Should only find uppercase constants, not base_url
        assert len(results) == 2
        constant_names = {r[1].node_name for r in results}
        assert "API_KEY" in constant_names
        assert "MAX_RETRIES" in constant_names
        assert "base_url" not in constant_names

    def test_should_parse_import_statements(self, python_parser):
        content = """import os
from pathlib import Path
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 2

        import_result = [r for r in results if r[1].node_name == "os"][0]
        assert import_result[1].node_type == "import"

        from_result = [r for r in results if r[1].node_name == "pathlib"][0]
        assert from_result[1].node_type == "import"

    def test_should_parse_decorated_class(self, python_parser):
        content = """@dataclass
class Point:
    x: int
    y: int
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Point"
        assert node_metadata.node_type == "class"
        assert "@dataclass" in node_metadata.extra["decorators"]

    def test_should_parse_function_with_type_hints(self, python_parser):
        content = """def add_numbers(a: int, b: int) -> int:
    return a + b
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "add_numbers"
        assert "int" in node_metadata.signature

    def test_should_handle_complex_code_with_multiple_elements(self, python_parser):
        content = '''"""Module docstring."""
import sys
from typing import List

MAX_SIZE = 100

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self):
        self.data = []
    
    @property
    def size(self):
        return len(self.data)
    
    async def process(self, items: List[str]) -> None:
        """Process items asynchronously."""
        for item in items:
            self.data.append(item)

def main():
    processor = DataProcessor()
    return processor
'''
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        # Should find: 2 imports, 1 constant, 1 class, 3 methods, 1 function
        assert len(results) >= 7

        # Verify constant
        constants = [r for r in results if r[1].node_type == "constant"]
        assert len(constants) == 1
        assert constants[0][1].node_name == "MAX_SIZE"

        # Verify class
        classes = [r for r in results if r[1].node_type == "class"]
        assert len(classes) == 1
        assert classes[0][1].node_name == "DataProcessor"

        # Verify methods
        methods = [r for r in results if r[1].node_type == "method"]
        assert len(methods) == 3
        method_names = {m[1].node_name for m in methods}
        assert "__init__" in method_names
        assert "size" in method_names
        assert "process" in method_names

        # Verify async method
        process_method = [m for m in methods if m[1].node_name == "process"][0]
        assert process_method[1].extra["is_async"] == "true"
        assert "@property" not in process_method[1].extra["decorators"]

        # Verify decorated method
        size_method = [m for m in methods if m[1].node_name == "size"][0]
        assert "@property" in size_method[1].extra["decorators"]

        # Verify module-level function
        functions = [r for r in results if r[1].node_type == "function"]
        assert len(functions) == 1
        assert functions[0][1].node_name == "main"

    def test_should_handle_empty_file(self, python_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))
        assert len(results) == 0

    def test_should_handle_file_with_only_comments(self, python_parser):
        content = """# This is a comment
# Another comment
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="comments.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))
        assert len(results) == 0

    def test_should_include_metadata_fields_from_document(self, python_parser):
        content = """def test_func():
    pass
"""
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/module.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/module.py"


class TestPythonParserInitialization:
    """Test PythonParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = PythonParser()
        assert parser.language == "python"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, python_parser):
        query = python_parser.query_str
        assert "function_definition" in query
        assert "class_definition" in query
        assert "decorated_definition" in query
        assert "import_statement" in query
        assert "import_from_statement" in query
        assert "assignment" in query


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_nested_classes(self, python_parser):
        content = """class Outer:
    class Inner:
        def method(self):
            pass
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        # Should find both classes and the method
        assert len(results) == 3

        # Verify nested method has correct parent scope
        method = [r for r in results if r[1].node_name == "method"][0]
        assert method[1].parent_scope == "Inner"

    def test_should_handle_multiline_decorators(self, python_parser):
        content = """@decorator_with_args(
    arg1="value1",
    arg2="value2"
)
def my_function():
    pass
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "my_function"
        # Decorator should be captured (even if multiline)
        assert node_metadata.extra["decorators"]

    def test_should_handle_lambda_functions(self, python_parser):
        # Lambda functions are typically not captured as they lack identifier nodes
        content = """lambda_func = lambda x: x * 2
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        # lambda_func is lowercase, should not be captured as constant
        assert len(results) == 0

    def test_should_parse_class_with_inheritance(self, python_parser):
        content = """class Child(Parent, Mixin):
    pass
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            ext=".py",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.py", content=content, metadata=metadata)

        results = list(python_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_name == "Child"
        assert node_metadata.node_type == "class"
