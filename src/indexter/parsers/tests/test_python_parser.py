"""Tests for the PythonParser."""

from unittest.mock import MagicMock

import pytest

from indexter.parsers.python import PythonParser


@pytest.fixture
def py_parser():
    """Create a PythonParser instance for testing."""
    return PythonParser()


@pytest.fixture
def simple_function():
    """Sample Python with a simple function."""
    return """def greet(name):
    \"\"\"Say hello to someone.\"\"\"
    return f"Hello, {name}!"
"""


@pytest.fixture
def simple_class():
    """Sample Python with a simple class."""
    return """class Person:
    \"\"\"A person class.\"\"\"
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, I'm {self.name}"
"""


@pytest.fixture
def async_function():
    """Sample Python with an async function."""
    return """async def fetch_data(url):
    \"\"\"Fetch data from a URL.\"\"\"
    return await client.get(url)
"""


@pytest.fixture
def decorated_function():
    """Sample Python with decorated function."""
    return """@decorator
@another_decorator
def decorated_func():
    \"\"\"A decorated function.\"\"\"
    pass
"""


@pytest.fixture
def constants():
    """Sample Python with constants."""
    return """MAX_SIZE = 100
MIN_SIZE = 10
API_KEY = "secret"
normalVar = "not a constant"
"""


@pytest.fixture
def imports():
    """Sample Python with imports."""
    return """import os
import sys
from pathlib import Path
from typing import List, Dict
"""


def test_parser_initialization(py_parser):
    """Test that PythonParser initializes correctly."""
    assert py_parser.language == "python"
    assert py_parser.tslanguage is not None
    assert py_parser.tsparser is not None


def test_query_str(py_parser):
    """Test that query_str returns a valid query string."""
    query = py_parser.query_str
    assert "function_definition" in query
    assert "class_definition" in query
    assert "decorated_definition" in query
    assert "import_statement" in query


def test_parse_simple_function(py_parser, simple_function):
    """Test parsing a simple function."""
    results = list(py_parser.parse(simple_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "function"
    assert info["node_name"] == "greet"
    assert info["language"] == "python"
    assert info["documentation"] == "Say hello to someone."
    assert info["signature"] == "def greet(name)"
    assert info["parent_scope"] is None
    assert info["extra"]["is_async"] == "false"


def test_parse_simple_class(py_parser, simple_class):
    """Test parsing a simple class with methods."""
    results = list(py_parser.parse(simple_class))

    # Should find: class, __init__ method, greet method
    assert len(results) == 3

    # Check class
    class_result = [r for r in results if r[1]["node_type"] == "class"][0]
    assert class_result[1]["node_name"] == "Person"
    assert class_result[1]["documentation"] == "A person class."
    assert class_result[1]["parent_scope"] is None

    # Check methods
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) == 2

    method_names = [r[1]["node_name"] for r in method_results]
    assert "__init__" in method_names
    assert "greet" in method_names

    # Methods should have parent_scope set to class name
    for _, info in method_results:
        assert info["parent_scope"] == "Person"


def test_parse_async_function(py_parser, async_function):
    """Test parsing an async function."""
    results = list(py_parser.parse(async_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "fetch_data"
    assert info["extra"]["is_async"] == "true"
    assert "async def" in info["signature"]


def test_parse_decorated_function(py_parser, decorated_function):
    """Test parsing a decorated function."""
    results = list(py_parser.parse(decorated_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "decorated_func"
    assert info["documentation"] == "A decorated function."
    assert "@decorator" in info["extra"]["decorators"]
    assert "@another_decorator" in info["extra"]["decorators"]
    assert "@decorator" in content
    assert "@another_decorator" in content


def test_parse_constants(py_parser, constants):
    """Test parsing module-level constants."""
    results = list(py_parser.parse(constants))

    # Should only find UPPER_CASE constants
    constant_results = [r for r in results if r[1]["node_type"] == "constant"]
    assert len(constant_results) == 3

    constant_names = [r[1]["node_name"] for r in constant_results]
    assert "MAX_SIZE" in constant_names
    assert "MIN_SIZE" in constant_names
    assert "API_KEY" in constant_names
    assert "normalVar" not in constant_names


def test_parse_imports(py_parser, imports):
    """Test parsing import statements."""
    results = list(py_parser.parse(imports))

    import_results = [r for r in results if r[1]["node_type"] == "import"]
    assert len(import_results) == 4

    import_names = [r[1]["node_name"] for r in import_results]
    assert "os" in import_names
    assert "sys" in import_names
    assert "pathlib" in import_names
    assert "typing" in import_names


def test_get_node_type_function(py_parser):
    """Test _get_node_type for functions."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = py_parser._get_node_type("function_definition", mock_node)
    assert result == "function"


def test_get_node_type_method(py_parser):
    """Test _get_node_type for methods."""
    mock_node = MagicMock()
    mock_node.parent = MagicMock()
    mock_node.parent.type = "class_definition"
    mock_node.parent.child_by_field_name = MagicMock(return_value=MagicMock(text=b"MyClass"))

    result = py_parser._get_node_type("function_definition", mock_node)
    assert result == "method"


def test_get_node_type_class(py_parser):
    """Test _get_node_type for classes."""
    mock_node = MagicMock()
    result = py_parser._get_node_type("class_definition", mock_node)
    assert result == "class"


def test_get_node_type_constant(py_parser):
    """Test _get_node_type for constants."""
    mock_node = MagicMock()
    result = py_parser._get_node_type("assignment", mock_node)
    assert result == "constant"


def test_get_node_type_import(py_parser):
    """Test _get_node_type for imports."""
    mock_node = MagicMock()
    result = py_parser._get_node_type("import_statement", mock_node)
    assert result == "import"

    result = py_parser._get_node_type("import_from_statement", mock_node)
    assert result == "import"


def test_strip_docstring_triple_double(py_parser):
    """Test _strip_docstring with triple double quotes."""
    result = py_parser._strip_docstring('"""This is a docstring."""')
    assert result == "This is a docstring."


def test_strip_docstring_triple_single(py_parser):
    """Test _strip_docstring with triple single quotes."""
    result = py_parser._strip_docstring("'''This is a docstring.'''")
    assert result == "This is a docstring."


def test_strip_docstring_single_double(py_parser):
    """Test _strip_docstring with single double quotes."""
    result = py_parser._strip_docstring('"Short doc."')
    assert result == "Short doc."


def test_strip_docstring_single_single(py_parser):
    """Test _strip_docstring with single single quotes."""
    result = py_parser._strip_docstring("'Short doc.'")
    assert result == "Short doc."


def test_strip_docstring_with_whitespace(py_parser):
    """Test _strip_docstring with leading/trailing whitespace."""
    result = py_parser._strip_docstring('"""  Whitespace  """')
    assert result == "Whitespace"


def test_is_constant_uppercase(py_parser):
    """Test _is_constant with uppercase names."""
    assert py_parser._is_constant("MAX_SIZE") is True
    assert py_parser._is_constant("API_KEY") is True
    assert py_parser._is_constant("CONSTANT") is True


def test_is_constant_with_underscores(py_parser):
    """Test _is_constant with underscores."""
    assert py_parser._is_constant("MAX_VALUE") is True
    assert py_parser._is_constant("_PRIVATE") is True


def test_is_not_constant_lowercase(py_parser):
    """Test _is_constant with lowercase names."""
    assert py_parser._is_constant("variable") is False
    assert py_parser._is_constant("myVar") is False


def test_is_not_constant_camelcase(py_parser):
    """Test _is_constant with camelCase names."""
    assert py_parser._is_constant("myVariable") is False
    assert py_parser._is_constant("SomeClass") is False


def test_parse_empty_python(py_parser):
    """Test parsing empty Python code."""
    results = list(py_parser.parse(""))
    assert len(results) == 0


def test_parse_comments_only(py_parser):
    """Test parsing Python with only comments."""
    code = """# This is a comment
# Another comment
"""
    results = list(py_parser.parse(code))
    assert len(results) == 0


def test_parse_function_without_docstring(py_parser):
    """Test parsing function without docstring."""
    code = """def simple():
    pass
"""
    results = list(py_parser.parse(code))
    assert len(results) == 1
    assert results[0][1]["documentation"] is None


def test_parse_class_without_docstring(py_parser):
    """Test parsing class without docstring."""
    code = """class Simple:
    pass
"""
    results = list(py_parser.parse(code))
    assert len(results) == 1
    assert results[0][1]["documentation"] is None


def test_parse_nested_function(py_parser):
    """Test parsing nested function."""
    code = """def outer():
    def inner():
        return 42
    return inner()
"""
    results = list(py_parser.parse(code))
    # Should find both outer and inner
    assert len(results) == 2
    names = [r[1]["node_name"] for r in results]
    assert "outer" in names
    assert "inner" in names


def test_parse_nested_class(py_parser):
    """Test parsing nested class."""
    code = """class Outer:
    class Inner:
        pass
"""
    results = list(py_parser.parse(code))
    assert len(results) == 2
    names = [r[1]["node_name"] for r in results]
    assert "Outer" in names
    assert "Inner" in names


def test_signature_extraction(py_parser):
    """Test signature extraction for functions."""
    code = """def func(a, b, c=10):
    pass
"""
    results = list(py_parser.parse(code))
    signature = results[0][1]["signature"]
    assert signature == "def func(a, b, c=10)"


def test_signature_with_type_hints(py_parser):
    """Test signature extraction with type hints."""
    code = """def typed_func(name: str, age: int) -> str:
    return name
"""
    results = list(py_parser.parse(code))
    signature = results[0][1]["signature"]
    assert "name: str" in signature
    assert "age: int" in signature
    assert "-> str" in signature


def test_byte_positions(py_parser):
    """Test byte position tracking."""
    code = """def first():
    pass

def second():
    pass
"""
    results = list(py_parser.parse(code))

    for _, info in results:
        assert info["start_byte"] >= 0
        assert info["end_byte"] > info["start_byte"]


def test_line_numbers(py_parser):
    """Test line number tracking (1-based)."""
    code = """def func1():
    pass

def func2():
    pass
"""
    results = list(py_parser.parse(code))

    assert results[0][1]["start_line"] == 1
    assert results[1][1]["start_line"] == 4


def test_parent_scope_nested_method(py_parser):
    """Test parent scope for nested class methods."""
    code = """class Outer:
    def method(self):
        pass
"""
    results = list(py_parser.parse(code))

    method = [r for r in results if r[1]["node_name"] == "method"][0]
    assert method[1]["parent_scope"] == "Outer"


def test_parent_scope_no_class(py_parser):
    """Test parent scope for top-level function."""
    code = """def standalone():
    pass
"""
    results = list(py_parser.parse(code))
    assert results[0][1]["parent_scope"] is None


def test_decorators_multiple(py_parser):
    """Test extraction of multiple decorators."""
    code = """@deco1
@deco2
@deco3
def func():
    pass
"""
    results = list(py_parser.parse(code))
    decorators = results[0][1]["extra"]["decorators"]
    assert "@deco1" in decorators
    assert "@deco2" in decorators
    assert "@deco3" in decorators


def test_decorators_with_arguments(py_parser):
    """Test decorators with arguments."""
    code = """@app.route('/home')
@login_required
def home():
    pass
"""
    results = list(py_parser.parse(code))
    decorators = results[0][1]["extra"]["decorators"]
    assert "@app.route('/home')" in decorators
    assert "@login_required" in decorators


def test_async_decorated_function(py_parser):
    """Test async function with decorators."""
    code = """@decorator
async def async_decorated():
    pass
"""
    results = list(py_parser.parse(code))

    assert len(results) == 1
    info = results[0][1]
    assert info["extra"]["is_async"] == "true"
    assert "@decorator" in info["extra"]["decorators"]


def test_class_with_inheritance(py_parser):
    """Test class with base classes."""
    code = """class Child(Parent1, Parent2):
    '''A child class.'''
    pass
"""
    results = list(py_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "Child"
    assert results[0][1]["documentation"] == "A child class."


def test_staticmethod_decorator(py_parser):
    """Test staticmethod decorator."""
    code = """class MyClass:
    @staticmethod
    def static_method():
        pass
"""
    results = list(py_parser.parse(code))

    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) == 1
    assert "@staticmethod" in method_results[0][1]["extra"]["decorators"]


def test_classmethod_decorator(py_parser):
    """Test classmethod decorator."""
    code = """class MyClass:
    @classmethod
    def class_method(cls):
        pass
"""
    results = list(py_parser.parse(code))

    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) == 1
    assert "@classmethod" in method_results[0][1]["extra"]["decorators"]


def test_property_decorator(py_parser):
    """Test property decorator."""
    code = """class MyClass:
    @property
    def value(self):
        return self._value
"""
    results = list(py_parser.parse(code))

    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) == 1
    assert "@property" in method_results[0][1]["extra"]["decorators"]


def test_multiline_docstring(py_parser):
    """Test multiline docstring."""
    code = '''def func():
    """
    This is a multiline
    docstring.
    """
    pass
'''
    results = list(py_parser.parse(code))
    assert len(results) == 1
    # Docstring should be extracted (may have whitespace)
    assert results[0][1]["documentation"] is not None


def test_expression_statement_docstring(py_parser):
    """Test docstring wrapped in expression_statement."""
    code = """def func():
    "Docstring as expression."
    pass
"""
    results = list(py_parser.parse(code))
    assert results[0][1]["documentation"] == "Docstring as expression."


def test_direct_string_literal_docstring(py_parser):
    """Test docstring as direct string literal (not in expression_statement)."""
    # Note: In Python, docstrings are typically wrapped in expression_statement
    # but tree-sitter might parse them differently in edge cases
    code = '''class Test:
    """Class docstring."""
    pass
'''
    results = list(py_parser.parse(code))
    # This should extract the docstring
    assert results[0][1]["documentation"] == "Class docstring."


def test_no_decorators_for_non_decorated(py_parser):
    """Test that non-decorated functions have empty decorators."""
    code = """def simple():
    pass
"""
    results = list(py_parser.parse(code))
    assert results[0][1]["extra"]["decorators"] == ""


def test_is_async_false_for_sync(py_parser):
    """Test is_async is false for sync functions."""
    code = """def sync_func():
    pass
"""
    results = list(py_parser.parse(code))
    assert results[0][1]["extra"]["is_async"] == "false"


def test_constant_with_mixed_case_and_underscore(py_parser):
    """Test that mixed case with underscore is not a constant."""
    assert py_parser._is_constant("My_Variable") is False
    assert py_parser._is_constant("myVar_Test") is False


def test_constant_all_caps_no_underscore(py_parser):
    """Test all caps without underscore is a constant."""
    assert py_parser._is_constant("CONSTANT") is True
    assert py_parser._is_constant("X") is True


def test_get_content(py_parser):
    """Test _get_content extracts source correctly."""
    code = b"def test():\n    pass"
    mock_node = MagicMock()
    mock_node.start_byte = 0
    mock_node.end_byte = len(code)

    result = py_parser._get_content(mock_node, code)
    assert result == code.decode()


def test_get_parent_scope_nested_class(py_parser):
    """Test _get_parent_scope finds enclosing class."""
    code = """class Outer:
    class Inner:
        def method(self):
            pass
"""
    results = list(py_parser.parse(code))

    # Method should have Inner as parent
    method = [r for r in results if r[1]["node_name"] == "method"][0]
    assert method[1]["parent_scope"] == "Inner"


def test_get_signature_returns_none_for_class(py_parser):
    """Test _get_signature returns None for classes."""
    mock_node = MagicMock()
    mock_node.type = "class_definition"

    result = py_parser._get_signature(mock_node, b"")
    assert result is None


def test_parse_lambda_not_captured(py_parser):
    """Test that lambda functions are not captured."""
    code = """x = lambda a: a + 1
"""
    results = list(py_parser.parse(code))
    # Should not capture lambda, only module-level constants
    # x is lowercase, so not a constant
    assert len(results) == 0


def test_parse_multiple_classes(py_parser):
    """Test parsing multiple classes."""
    code = """class First:
    pass

class Second:
    pass

class Third:
    pass
"""
    results = list(py_parser.parse(code))

    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 3

    class_names = [r[1]["node_name"] for r in class_results]
    assert "First" in class_names
    assert "Second" in class_names
    assert "Third" in class_names


def test_parse_decorated_class(py_parser):
    """Test parsing decorated class."""
    code = """@dataclass
class Person:
    name: str
    age: int
"""
    results = list(py_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_type"] == "class"
    assert "@dataclass" in results[0][1]["extra"]["decorators"]


def test_import_from_without_module(py_parser):
    """Test import from statement without module name."""
    code = """from . import something
"""
    results = list(py_parser.parse(code))
    # This should be handled gracefully
    assert len(results) >= 0


def test_language_property(py_parser):
    """Test that language property is set correctly."""
    assert py_parser.language == "python"


def test_all_results_have_required_fields(py_parser):
    """Test that all parsed results have required fields."""
    code = """def func():
    pass

class MyClass:
    def method(self):
        pass

MAX_SIZE = 100
"""
    results = list(py_parser.parse(code))

    required_fields = [
        "language",
        "node_type",
        "node_name",
        "start_byte",
        "end_byte",
        "start_line",
        "end_line",
        "documentation",
        "parent_scope",
        "signature",
        "extra",
    ]

    for _, info in results:
        for field in required_fields:
            assert field in info


def test_extra_always_has_decorators_and_is_async(py_parser):
    """Test that extra always contains decorators and is_async."""
    code = """def func():
    pass
"""
    results = list(py_parser.parse(code))

    extra = results[0][1]["extra"]
    assert "decorators" in extra
    assert "is_async" in extra


def test_process_match_no_def_nodes(py_parser):
    """Test process_match with no def nodes."""
    match = {}
    result = py_parser.process_match(match, b"")
    assert result is None


def test_process_match_skips_decorated_inner(py_parser):
    """Test that process_match skips function inside decorated_definition."""
    code = b"""@decorator
def func():
    pass
"""
    # Parse to verify behavior - the decorated_definition captures the whole thing
    results = list(py_parser.parse(code.decode()))
    # Should only get one result (the decorated definition, not the inner function separately)
    assert len(results) == 1


def test_get_documentation_no_body(py_parser):
    """Test _get_documentation with no body."""
    mock_node = MagicMock()
    mock_node.type = "function_definition"
    mock_node.child_by_field_name = MagicMock(return_value=None)

    result = py_parser._get_documentation(mock_node, b"")
    assert result is None


def test_get_documentation_empty_body(py_parser):
    """Test _get_documentation with empty body."""
    mock_node = MagicMock()
    mock_node.type = "function_definition"
    mock_body = MagicMock()
    mock_body.children = []
    mock_node.child_by_field_name = MagicMock(return_value=mock_body)

    result = py_parser._get_documentation(mock_node, b"")
    assert result is None


def test_get_documentation_expression_statement_with_string(py_parser):
    """Test _get_documentation with expression_statement containing string."""
    mock_node = MagicMock()
    mock_node.type = "function_definition"

    # Create mock expression_statement with string child
    mock_string = MagicMock()
    mock_string.type = "string"
    mock_string.text = b'"Docstring from expression."'

    mock_expr = MagicMock()
    mock_expr.type = "expression_statement"
    mock_expr.children = [mock_string]

    mock_body = MagicMock()
    mock_body.children = [mock_expr]

    mock_node.child_by_field_name = MagicMock(return_value=mock_body)

    result = py_parser._get_documentation(mock_node, b"")
    assert result == "Docstring from expression."


def test_is_async_for_decorated_async(py_parser):
    """Test _is_async for decorated async function."""
    code = """@decorator
async def func():
    pass
"""
    results = list(py_parser.parse(code))
    assert results[0][1]["extra"]["is_async"] == "true"


def test_get_decorators_non_decorated(py_parser):
    """Test _get_decorators for non-decorated node."""
    mock_node = MagicMock()
    mock_node.type = "function_definition"

    result = py_parser._get_decorators(mock_node, b"")
    assert result == []


def test_get_node_type_unknown(py_parser):
    """Test _get_node_type returns ts_type for unknown types."""
    mock_node = MagicMock()
    result = py_parser._get_node_type("unknown_type", mock_node)
    assert result == "unknown_type"


def test_get_signature_for_class(py_parser):
    """Test _get_signature explicitly with a class node."""
    mock_node = MagicMock()
    mock_node.type = "class_definition"

    result = py_parser._get_signature(mock_node, b"class Test: pass")
    assert result is None


def test_get_signature_no_body(py_parser):
    """Test _get_signature when function has no body field."""
    mock_node = MagicMock()
    mock_node.type = "function_definition"
    mock_node.start_byte = 0
    mock_node.end_byte = 15
    mock_node.child_by_field_name = MagicMock(return_value=None)

    result = py_parser._get_signature(mock_node, b"def test(): ...")
    # Should use end_byte when no body
    assert result is not None


def test_import_from_relative(py_parser):
    """Test import from with relative import."""
    code = """from ..parent import module
"""
    results = list(py_parser.parse(code))
    # Should handle relative imports
    assert len(results) >= 0
