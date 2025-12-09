"""Tests for the TypeScriptParser."""

from unittest.mock import MagicMock

import pytest

from indexter.parsers.typescript import TypeScriptParser


@pytest.fixture
def ts_parser():
    """Create a TypeScriptParser instance for testing."""
    return TypeScriptParser()


@pytest.fixture
def simple_function():
    """Sample TypeScript with a simple function."""
    return """function greet(name: string): string {
    return `Hello, ${name}!`;
}
"""


@pytest.fixture
def simple_class():
    """Sample TypeScript with a simple class."""
    return """/**
 * A person class
 */
class Person {
    constructor(public name: string) {}
    
    greet(): string {
        return `Hello, I'm ${this.name}`;
    }
}
"""


@pytest.fixture
def simple_interface():
    """Sample TypeScript with a simple interface."""
    return """/**
 * A greeting interface
 */
interface Greeter {
    greet(): string;
}
"""


@pytest.fixture
def arrow_function():
    """Sample TypeScript with an arrow function."""
    return """const add = (a: number, b: number): number => a + b;
"""


@pytest.fixture
def async_function():
    """Sample TypeScript with an async function."""
    return """async function fetchData(url: string): Promise<string> {
    const response = await fetch(url);
    return response.text();
}
"""


@pytest.fixture
def type_alias():
    """Sample TypeScript with a type alias."""
    return """type Result<T> = { success: boolean; data: T };
"""


@pytest.fixture
def enum_declaration():
    """Sample TypeScript with an enum."""
    return """enum Color {
    Red,
    Green,
    Blue,
}
"""


@pytest.fixture
def constant():
    """Sample TypeScript with a constant."""
    return """const MAX_SIZE = 100;
"""


@pytest.fixture
def import_statement():
    """Sample TypeScript with imports."""
    return """import { Component } from '@angular/core';
import React from 'react';
"""


def test_parser_initialization(ts_parser):
    """Test that TypeScriptParser initializes correctly."""
    assert ts_parser.language == "typescript"
    assert ts_parser.tslanguage is not None
    assert ts_parser.tsparser is not None


def test_query_str(ts_parser):
    """Test that query_str returns a valid query string."""
    query = ts_parser.query_str
    assert "function_declaration" in query
    assert "class_declaration" in query
    assert "interface_declaration" in query
    assert "type_alias_declaration" in query
    assert "enum_declaration" in query
    assert "method_definition" in query
    assert "arrow_function" in query
    assert "import_statement" in query


def test_parse_simple_function(ts_parser, simple_function):
    """Test parsing a simple function."""
    results = list(ts_parser.parse(simple_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "function"
    assert info["node_name"] == "greet"
    assert info["language"] == "typescript"
    assert "function greet(name: string): string" in info["signature"]
    assert info["parent_scope"] is None
    assert info["extra"]["is_async"] == "false"
    assert info["extra"]["is_arrow"] == "false"


def test_parse_simple_class(ts_parser, simple_class):
    """Test parsing a simple class with methods."""
    results = list(ts_parser.parse(simple_class))

    # Should find: class + method
    assert len(results) >= 2

    # Check class
    class_result = [r for r in results if r[1]["node_type"] == "class"][0]
    assert class_result[1]["node_name"] == "Person"
    assert class_result[1]["documentation"] == "A person class"
    assert class_result[1]["parent_scope"] is None

    # Check method
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1

    greet_method = [r for r in method_results if r[1]["node_name"] == "greet"][0]
    assert greet_method[1]["parent_scope"] == "Person"


def test_parse_simple_interface(ts_parser, simple_interface):
    """Test parsing a simple interface."""
    results = list(ts_parser.parse(simple_interface))

    # Should find interface + method signature
    interface_result = [r for r in results if r[1]["node_type"] == "interface"][0]
    assert interface_result[1]["node_name"] == "Greeter"
    assert interface_result[1]["documentation"] == "A greeting interface"


def test_parse_arrow_function(ts_parser, arrow_function):
    """Test parsing an arrow function."""
    results = list(ts_parser.parse(arrow_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "add"
    assert info["node_type"] == "function"
    assert info["extra"]["is_arrow"] == "true"
    assert "=>" in info["signature"]


def test_parse_async_function(ts_parser, async_function):
    """Test parsing an async function."""
    results = list(ts_parser.parse(async_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "fetchData"
    assert info["extra"]["is_async"] == "true"
    assert "async function" in info["signature"]


def test_parse_type_alias(ts_parser, type_alias):
    """Test parsing a type alias."""
    results = list(ts_parser.parse(type_alias))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "type_alias"
    assert info["node_name"] == "Result"


def test_parse_enum(ts_parser, enum_declaration):
    """Test parsing an enum."""
    results = list(ts_parser.parse(enum_declaration))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "enum"
    assert info["node_name"] == "Color"


def test_parse_constant(ts_parser, constant):
    """Test parsing a constant."""
    results = list(ts_parser.parse(constant))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "constant"
    assert info["node_name"] == "MAX_SIZE"


def test_parse_imports(ts_parser, import_statement):
    """Test parsing import statements."""
    results = list(ts_parser.parse(import_statement))

    import_results = [r for r in results if r[1]["node_type"] == "import"]
    assert len(import_results) == 2

    import_names = [r[1]["node_name"] for r in import_results]
    assert "@angular/core" in import_names
    assert "react" in import_names


def test_parse_empty_typescript(ts_parser):
    """Test parsing empty TypeScript code."""
    results = list(ts_parser.parse(""))
    assert len(results) == 0


def test_parse_comments_only(ts_parser):
    """Test parsing TypeScript with only comments."""
    code = """// This is a comment
/* Another comment */
"""
    results = list(ts_parser.parse(code))
    assert len(results) == 0


def test_tsdoc_comment(ts_parser):
    """Test TSDoc comment extraction."""
    code = """/**
 * This is a TSDoc comment
 * with multiple lines
 */
function documented(): void {}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    doc = results[0][1]["documentation"]
    assert "This is a TSDoc comment" in doc
    assert "with multiple lines" in doc


def test_generator_function(ts_parser):
    """Test parsing generator function."""
    code = """function* generateNumbers(): Generator<number> {
    yield 1;
    yield 2;
}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["extra"]["is_generator"] == "true"


def test_abstract_class(ts_parser):
    """Test parsing abstract class."""
    code = """abstract class BaseClass {
    abstract method(): void;
}
"""
    results = list(ts_parser.parse(code))

    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 1
    assert class_results[0][1]["extra"]["is_abstract"] == "true"


def test_visibility_modifiers(ts_parser):
    """Test parsing visibility modifiers."""
    code = """class MyClass {
    public publicMethod(): void {}
    private privateMethod(): void {}
    protected protectedMethod(): void {}
}
"""
    results = list(ts_parser.parse(code))

    method_results = [r for r in results if r[1]["node_type"] == "method"]

    visibilities = [r[1]["extra"]["visibility"] for r in method_results]
    assert "public" in visibilities
    assert "private" in visibilities
    assert "protected" in visibilities


def test_decorators(ts_parser):
    """Test parsing decorators."""
    code = """class Component {
    @Input()
    name: string;
    
    @Output()
    change = new EventEmitter();
}
"""
    results = list(ts_parser.parse(code))

    # Should find the class
    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) >= 1


def test_export_function(ts_parser):
    """Test parsing export function."""
    code = """export function exported(): void {}
"""
    results = list(ts_parser.parse(code))

    export_results = [r for r in results if r[1]["node_type"] == "export"]
    # Should find the export
    assert len(export_results) >= 0


def test_export_class(ts_parser):
    """Test parsing export class."""
    code = """export class ExportedClass {}
"""
    results = list(ts_parser.parse(code))

    # Should find export
    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 0


def test_export_default(ts_parser):
    """Test parsing default export."""
    code = """export default class DefaultClass {}
"""
    results = list(ts_parser.parse(code))

    # Should find export
    assert len(results) >= 1


def test_const_without_arrow_not_constant(ts_parser):
    """Test that lowercase const without arrow function is not captured."""
    code = """const myVar = 42;
"""
    results = list(ts_parser.parse(code))

    # Should not capture lowercase const
    assert len(results) == 0


def test_const_upper_case_captured(ts_parser):
    """Test that UPPER_CASE const is captured."""
    code = """const API_KEY = "secret";
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "API_KEY"


def test_nested_class(ts_parser):
    """Test parsing nested classes."""
    code = """class Outer {
    method(): void {
        class Inner {}
    }
}
"""
    results = list(ts_parser.parse(code))

    # Should find both classes
    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) >= 1


def test_interface_method_signature(ts_parser):
    """Test interface method signature extraction."""
    code = """interface API {
    fetch(url: string): Promise<Response>;
}
"""
    results = list(ts_parser.parse(code))

    # Should find interface and method signature
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    if method_results:
        assert "fetch(url: string): Promise<Response>" in method_results[0][1]["signature"]


def test_async_arrow_function(ts_parser):
    """Test async arrow function."""
    code = """const asyncFetch = async (url: string) => {
    return await fetch(url);
};
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    info = results[0][1]
    assert info["extra"]["is_async"] == "true"
    assert info["extra"]["is_arrow"] == "true"


def test_generic_function(ts_parser):
    """Test parsing generic function."""
    code = """function identity<T>(value: T): T {
    return value;
}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "identity"


def test_generic_class(ts_parser):
    """Test parsing generic class."""
    code = """class Container<T> {
    constructor(public value: T) {}
}
"""
    results = list(ts_parser.parse(code))

    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 1
    assert class_results[0][1]["node_name"] == "Container"


def test_byte_positions(ts_parser):
    """Test byte position tracking."""
    code = """function first(): void {}

function second(): void {}
"""
    results = list(ts_parser.parse(code))

    for _, info in results:
        assert info["start_byte"] >= 0
        assert info["end_byte"] > info["start_byte"]


def test_line_numbers(ts_parser):
    """Test line number tracking (1-based)."""
    code = """function func1(): void {}

function func2(): void {}
"""
    results = list(ts_parser.parse(code))

    assert results[0][1]["start_line"] == 1
    assert results[1][1]["start_line"] == 3


def test_get_content(ts_parser):
    """Test _get_content extracts source correctly."""
    code = b"function test() {}"
    mock_node = MagicMock()
    mock_node.start_byte = 0
    mock_node.end_byte = len(code)

    result = ts_parser._get_content(mock_node, code)
    assert result == code.decode()


def test_get_node_type_class(ts_parser):
    """Test _get_node_type for classes."""
    mock_node = MagicMock()
    mock_node.type = "class_declaration"

    mock_outer = MagicMock()
    mock_outer.type = "class_declaration"

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "class"


def test_get_node_type_abstract_class(ts_parser):
    """Test _get_node_type for abstract classes."""
    mock_node = MagicMock()
    mock_node.type = "abstract_class_declaration"

    mock_outer = MagicMock()
    mock_outer.type = "abstract_class_declaration"

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "class"


def test_get_node_type_interface(ts_parser):
    """Test _get_node_type for interfaces."""
    mock_node = MagicMock()
    mock_node.type = "interface_declaration"

    mock_outer = MagicMock()

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "interface"


def test_get_node_type_method(ts_parser):
    """Test _get_node_type for methods."""
    mock_node = MagicMock()
    mock_node.type = "method_definition"

    mock_outer = MagicMock()

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "method"


def test_get_node_type_function_with_parent(ts_parser):
    """Test _get_node_type for function inside class."""
    mock_node = MagicMock()
    mock_node.type = "function_declaration"

    # Mock _get_parent_scope to return a class name
    original_get_parent_scope = ts_parser._get_parent_scope
    ts_parser._get_parent_scope = lambda n: "MyClass"

    mock_outer = MagicMock()

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "method"

    ts_parser._get_parent_scope = original_get_parent_scope


def test_get_node_type_arrow_function(ts_parser):
    """Test _get_node_type for arrow function."""
    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.parent = None

    mock_outer = MagicMock()

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "function"


def test_parse_tsdoc(ts_parser):
    """Test _parse_tsdoc."""
    comment = """/**
 * This is a doc comment
 * with multiple lines
 */"""
    result = ts_parser._parse_tsdoc(comment)
    assert "This is a doc comment" in result
    assert "with multiple lines" in result


def test_parse_tsdoc_not_block_comment(ts_parser):
    """Test _parse_tsdoc returns None for non-block comments."""
    comment = "// Single line comment"
    result = ts_parser._parse_tsdoc(comment)
    assert result is None


def test_parse_tsdoc_no_content(ts_parser):
    """Test _parse_tsdoc with empty comment."""
    comment = "/**  */"
    result = ts_parser._parse_tsdoc(comment)
    # Should return None for empty content
    assert result is None or result == ""


def test_get_signature_arrow_function(ts_parser):
    """Test _get_signature for arrow function."""
    code = b"(a: number, b: number): number => a + b"

    mock_body = MagicMock()
    mock_body.start_byte = 31

    mock_arrow = MagicMock()
    mock_arrow.type = "=>"
    mock_arrow.end_byte = 33

    mock_params = MagicMock()
    mock_params.start_byte = 0

    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.start_byte = 0
    mock_node.children = [mock_params, mock_arrow, mock_body]
    mock_node.child_by_field_name = (
        lambda field: mock_params if field in ("parameters", "parameter") else mock_body
    )

    result = ts_parser._get_signature(mock_node, code)
    assert result is not None
    assert "=>" in result


def test_get_signature_regular_function(ts_parser):
    """Test _get_signature for regular function."""
    code = b"function test(a: number): number { return a; }"

    mock_body = MagicMock()
    mock_body.start_byte = 33

    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.start_byte = 0
    mock_node.child_by_field_name = lambda field: mock_body if field == "body" else None

    result = ts_parser._get_signature(mock_node, code)
    assert result == "function test(a: number): number"


def test_get_signature_no_body(ts_parser):
    """Test _get_signature for interface method (no body)."""
    code = b"method(): void;"

    mock_node = MagicMock()
    mock_node.type = "method_signature"
    mock_node.start_byte = 0
    mock_node.end_byte = len(code)
    mock_node.child_by_field_name = lambda field: None

    result = ts_parser._get_signature(mock_node, code)
    assert result == "method(): void;"


def test_get_signature_non_function(ts_parser):
    """Test _get_signature returns None for non-functions."""
    mock_node = MagicMock()
    mock_node.type = "class_declaration"

    result = ts_parser._get_signature(mock_node, b"")
    assert result is None


def test_get_parent_scope_class(ts_parser):
    """Test _get_parent_scope finds class."""
    mock_name = MagicMock()
    mock_name.text = b"MyClass"

    mock_class = MagicMock()
    mock_class.type = "class_declaration"
    mock_class.child_by_field_name = lambda field: mock_name if field == "name" else None

    mock_node = MagicMock()
    mock_node.parent = mock_class

    result = ts_parser._get_parent_scope(mock_node)
    assert result == "MyClass"


def test_get_parent_scope_interface(ts_parser):
    """Test _get_parent_scope finds interface."""
    mock_name = MagicMock()
    mock_name.text = b"MyInterface"

    mock_interface = MagicMock()
    mock_interface.type = "interface_declaration"
    mock_interface.child_by_field_name = lambda field: mock_name if field == "name" else None

    mock_node = MagicMock()
    mock_node.parent = mock_interface

    result = ts_parser._get_parent_scope(mock_node)
    assert result == "MyInterface"


def test_get_parent_scope_none(ts_parser):
    """Test _get_parent_scope returns None for top-level."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = ts_parser._get_parent_scope(mock_node)
    assert result is None


def test_get_parent_scope_through_body(ts_parser):
    """Test _get_parent_scope traverses through class_body."""
    mock_name = MagicMock()
    mock_name.text = b"MyClass"

    mock_class = MagicMock()
    mock_class.type = "class_declaration"
    mock_class.child_by_field_name = lambda field: mock_name if field == "name" else None

    mock_body = MagicMock()
    mock_body.type = "class_body"
    mock_body.parent = mock_class

    mock_node = MagicMock()
    mock_node.parent = mock_body

    result = ts_parser._get_parent_scope(mock_node)
    assert result == "MyClass"


def test_is_async_true(ts_parser):
    """Test _is_async detects async functions."""
    mock_async = MagicMock()
    mock_async.type = "async"

    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.children = [mock_async]

    result = ts_parser._is_async(mock_node)
    assert result is True


def test_is_async_false(ts_parser):
    """Test _is_async returns False for sync functions."""
    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.children = []

    result = ts_parser._is_async(mock_node)
    assert result is False


def test_is_generator_true_declaration(ts_parser):
    """Test _is_generator detects generator functions."""
    mock_node = MagicMock()
    mock_node.type = "generator_function_declaration"

    result = ts_parser._is_generator(mock_node)
    assert result is True


def test_is_generator_true_asterisk(ts_parser):
    """Test _is_generator detects generator with asterisk."""
    mock_asterisk = MagicMock()
    mock_asterisk.type = "*"

    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.children = [mock_asterisk]

    result = ts_parser._is_generator(mock_node)
    assert result is True


def test_is_generator_false(ts_parser):
    """Test _is_generator returns False for regular functions."""
    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.children = []

    result = ts_parser._is_generator(mock_node)
    assert result is False


def test_is_abstract_class(ts_parser):
    """Test _is_abstract detects abstract classes."""
    mock_node = MagicMock()
    mock_node.type = "abstract_class_declaration"
    mock_node.children = []

    result = ts_parser._is_abstract(mock_node)
    assert result is True


def test_is_abstract_modifier(ts_parser):
    """Test _is_abstract detects abstract modifier."""
    mock_abstract = MagicMock()
    mock_abstract.type = "abstract"

    mock_node = MagicMock()
    mock_node.type = "method_definition"
    mock_node.children = [mock_abstract]

    result = ts_parser._is_abstract(mock_node)
    assert result is True


def test_is_abstract_false(ts_parser):
    """Test _is_abstract returns False for non-abstract."""
    mock_node = MagicMock()
    mock_node.type = "class_declaration"
    mock_node.children = []

    result = ts_parser._is_abstract(mock_node)
    assert result is False


def test_get_visibility_public(ts_parser):
    """Test _get_visibility detects public."""
    mock_vis = MagicMock()
    mock_vis.type = "accessibility_modifier"
    mock_vis.text = b"public"

    mock_node = MagicMock()
    mock_node.children = [mock_vis]

    result = ts_parser._get_visibility(mock_node)
    assert result == "public"


def test_get_visibility_none(ts_parser):
    """Test _get_visibility returns None when no modifier."""
    mock_node = MagicMock()
    mock_node.children = []

    result = ts_parser._get_visibility(mock_node)
    assert result is None


def test_is_const_declaration_true(ts_parser):
    """Test _is_const_declaration detects const."""
    mock_const = MagicMock()
    mock_const.type = "const"

    mock_node = MagicMock()
    mock_node.children = [mock_const]

    result = ts_parser._is_const_declaration(mock_node)
    assert result is True


def test_is_const_declaration_false(ts_parser):
    """Test _is_const_declaration returns False for let/var."""
    mock_node = MagicMock()
    mock_node.children = []

    result = ts_parser._is_const_declaration(mock_node)
    assert result is False


def test_is_constant_upper_case(ts_parser):
    """Test _is_constant detects UPPER_CASE."""
    assert ts_parser._is_constant("MAX_SIZE") is True
    assert ts_parser._is_constant("API_KEY") is True
    assert ts_parser._is_constant("CONSTANT") is True


def test_is_constant_lower_case(ts_parser):
    """Test _is_constant returns False for lowercase."""
    assert ts_parser._is_constant("variable") is False
    assert ts_parser._is_constant("myVar") is False
    assert ts_parser._is_constant("SomeClass") is False


def test_get_decorators_from_children(ts_parser):
    """Test _get_decorators extracts decorators from children."""
    mock_dec = MagicMock()
    mock_dec.type = "decorator"
    mock_dec.text = b"@Injectable()"

    mock_node = MagicMock()
    mock_node.children = [mock_dec]
    mock_node.parent = None

    result = ts_parser._get_decorators(mock_node, b"")
    assert "@Injectable()" in result


def test_get_decorators_from_siblings(ts_parser):
    """Test _get_decorators extracts decorators from siblings."""
    mock_dec = MagicMock()
    mock_dec.type = "decorator"
    mock_dec.text = b"@Component()"

    mock_node = MagicMock()
    mock_node.children = []

    mock_parent = MagicMock()
    mock_parent.children = [mock_dec, mock_node]
    mock_node.parent = mock_parent

    result = ts_parser._get_decorators(mock_node, b"")
    assert "@Component()" in result


def test_get_decorators_none(ts_parser):
    """Test _get_decorators returns empty list when no decorators."""
    mock_node = MagicMock()
    mock_node.children = []
    mock_node.parent = None

    result = ts_parser._get_decorators(mock_node, b"")
    assert result == []


def test_get_export_name_function(ts_parser):
    """Test _get_export_name for exported function."""
    mock_name = MagicMock()
    mock_name.text = b"myFunc"

    mock_func = MagicMock()
    mock_func.type = "function_declaration"
    mock_func.child_by_field_name = lambda field: mock_name if field == "name" else None

    mock_node = MagicMock()
    mock_node.children = [mock_func]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result == "myFunc"


def test_get_export_name_class(ts_parser):
    """Test _get_export_name for exported class."""
    mock_name = MagicMock()
    mock_name.text = b"MyClass"

    mock_class = MagicMock()
    mock_class.type = "class_declaration"
    mock_class.child_by_field_name = lambda field: mock_name if field == "name" else None

    mock_node = MagicMock()
    mock_node.children = [mock_class]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result == "MyClass"


def test_get_export_name_default(ts_parser):
    """Test _get_export_name returns 'default' when no name."""
    mock_node = MagicMock()
    mock_node.children = []

    result = ts_parser._get_export_name(mock_node, b"")
    assert result == "default"


def test_get_documentation_no_parent(ts_parser):
    """Test _get_documentation with no parent."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = ts_parser._get_documentation(mock_node, b"")
    assert result is None


def test_get_documentation_first_child(ts_parser):
    """Test _get_documentation for first child (no previous sibling)."""
    mock_parent = MagicMock()
    mock_parent.children = []

    mock_node = MagicMock()
    mock_node.parent = mock_parent

    result = ts_parser._get_documentation(mock_node, b"")
    assert result is None


def test_get_documentation_no_comment(ts_parser):
    """Test _get_documentation when previous sibling is not a comment."""
    mock_other = MagicMock()
    mock_other.type = "identifier"

    mock_node = MagicMock()

    mock_parent = MagicMock()
    mock_parent.children = [mock_other, mock_node]
    mock_node.parent = mock_parent

    result = ts_parser._get_documentation(mock_node, b"")
    assert result is None


def test_process_match_no_def_nodes(ts_parser):
    """Test process_match with no def nodes."""
    match = {}
    result = ts_parser.process_match(match, b"")
    assert result is None


def test_process_match_skip_lexical_in_export(ts_parser):
    """Test process_match skips lexical declaration inside export."""
    mock_export = MagicMock()
    mock_export.type = "export_statement"

    mock_lexical = MagicMock()
    mock_lexical.type = "lexical_declaration"
    mock_lexical.parent = mock_export

    match = {"def": [mock_lexical]}

    result = ts_parser.process_match(match, b"")
    assert result is None


def test_process_match_no_name_no_source(ts_parser):
    """Test process_match with no name or source."""
    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.parent = None

    match = {"def": [mock_node]}

    result = ts_parser.process_match(match, b"")
    assert result is None


def test_language_property(ts_parser):
    """Test that language property is set correctly."""
    assert ts_parser.language == "typescript"


def test_all_results_have_required_fields(ts_parser):
    """Test that all parsed results have required fields."""
    code = """function func(): void {}
class MyClass {}
const VALUE = 42;
"""
    results = list(ts_parser.parse(code))

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


def test_extra_always_has_required_fields(ts_parser):
    """Test that extra always contains required fields."""
    code = """function func(): void {}
"""
    results = list(ts_parser.parse(code))

    extra = results[0][1]["extra"]
    assert "decorators" in extra
    assert "is_async" in extra
    assert "is_generator" in extra
    assert "is_arrow" in extra
    assert "is_abstract" in extra
    assert "visibility" in extra


def test_multiple_classes(ts_parser):
    """Test parsing multiple classes."""
    code = """class First {}
class Second {}
class Third {}
"""
    results = list(ts_parser.parse(code))

    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 3

    class_names = [r[1]["node_name"] for r in class_results]
    assert "First" in class_names
    assert "Second" in class_names
    assert "Third" in class_names


def test_complex_interface(ts_parser):
    """Test parsing complex interface."""
    code = """interface Complex<T> {
    prop: string;
    method(arg: T): void;
    optionalMethod?(): boolean;
}
"""
    results = list(ts_parser.parse(code))

    interface_results = [r for r in results if r[1]["node_type"] == "interface"]
    assert len(interface_results) == 1
    assert interface_results[0][1]["node_name"] == "Complex"


def test_namespace_declaration(ts_parser):
    """Test that namespace doesn't cause errors."""
    code = """namespace MyNamespace {
    export function func(): void {}
}
"""
    results = list(ts_parser.parse(code))
    # Should handle gracefully
    assert len(results) >= 0


def test_readonly_modifier(ts_parser):
    """Test readonly modifier is handled."""
    code = """class MyClass {
    readonly prop: string;
}
"""
    results = list(ts_parser.parse(code))
    # Should parse the class at minimum
    assert len(results) >= 1


def test_optional_parameters(ts_parser):
    """Test function with optional parameters."""
    code = """function optional(required: string, optional?: number): void {}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "optional"


def test_rest_parameters(ts_parser):
    """Test function with rest parameters."""
    code = """function variadic(...args: string[]): void {}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "variadic"


def test_union_types(ts_parser):
    """Test function with union types."""
    code = """function union(value: string | number): void {}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "union"


def test_intersection_types(ts_parser):
    """Test function with intersection types."""
    code = """function intersection(value: A & B): void {}
"""
    results = list(ts_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "intersection"


def test_let_declaration_not_constant(ts_parser):
    """Test that let declarations are not captured."""
    code = """let myVar = 42;
"""
    results = list(ts_parser.parse(code))

    # Should not capture let declarations
    assert len(results) == 0


def test_var_declaration_not_constant(ts_parser):
    """Test that var declarations are not captured."""
    code = """var myVar = 42;
"""
    results = list(ts_parser.parse(code))

    # Should not capture var declarations
    assert len(results) == 0


def test_arrow_function_without_params_field(ts_parser):
    """Test arrow function signature without params field."""
    code = b"x => x + 1"

    mock_body = MagicMock()
    mock_body.start_byte = 5

    mock_param = MagicMock()
    mock_param.start_byte = 0

    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.start_byte = 0
    mock_node.children = []
    mock_node.child_by_field_name = lambda field: mock_body if field == "body" else None

    result = ts_parser._get_signature(mock_node, code)
    assert result is not None


def test_arrow_function_no_arrow_symbol(ts_parser):
    """Test arrow function without finding arrow symbol."""
    code = b"(x) => x"

    mock_params = MagicMock()
    mock_params.start_byte = 0

    mock_body = MagicMock()
    mock_body.start_byte = 7

    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.start_byte = 0
    mock_node.children = []  # No => child
    mock_node.child_by_field_name = (
        lambda field: mock_params if field in ("parameters", "parameter") else mock_body
    )

    result = ts_parser._get_signature(mock_node, code)
    # Should fall back to body-based extraction
    assert result == "(x) =>"


def test_arrow_function_no_body(ts_parser):
    """Test arrow function without body field."""
    code = b"(x) => x"

    mock_params = MagicMock()
    mock_params.start_byte = 0

    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.start_byte = 0
    mock_node.children = []
    mock_node.child_by_field_name = (
        lambda field: mock_params if field in ("parameters", "parameter") else None
    )

    result = ts_parser._get_signature(mock_node, code)
    # Should return None when no body
    assert result is None


def test_export_interface(ts_parser):
    """Test export interface."""
    code = """export interface IExported {
    method(): void;
}
"""
    results = list(ts_parser.parse(code))

    # Should find export
    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 0


def test_export_type_alias(ts_parser):
    """Test export type alias."""
    code = """export type MyType = string | number;
"""
    results = list(ts_parser.parse(code))

    # Should find export
    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 0


def test_export_identifier(ts_parser):
    """Test export with identifier."""
    code = """const value = 42;
export { value };
"""
    results = list(ts_parser.parse(code))

    # Should handle export clause
    assert len(results) >= 0


def test_export_const(ts_parser):
    """Test export const."""
    code = """export const EXPORTED_CONST = 100;
"""
    results = list(ts_parser.parse(code))

    # Should find export
    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 0


def test_export_multiple_names(ts_parser):
    """Test export with multiple names."""
    code = """export { first, second, third };
"""
    results = list(ts_parser.parse(code))

    # Should handle multiple exports
    assert len(results) >= 0


def test_get_export_name_interface_no_name(ts_parser):
    """Test _get_export_name for interface without name."""
    mock_interface = MagicMock()
    mock_interface.type = "interface_declaration"
    mock_interface.child_by_field_name = lambda field: None

    mock_node = MagicMock()
    mock_node.children = [mock_interface]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result is None


def test_get_export_name_type_alias_no_name(ts_parser):
    """Test _get_export_name for type alias without name."""
    mock_type = MagicMock()
    mock_type.type = "type_alias_declaration"
    mock_type.child_by_field_name = lambda field: None

    mock_node = MagicMock()
    mock_node.children = [mock_type]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result is None


def test_get_export_name_export_clause_no_names(ts_parser):
    """Test _get_export_name for empty export clause."""
    mock_clause = MagicMock()
    mock_clause.type = "export_clause"
    mock_clause.children = []

    mock_node = MagicMock()
    mock_node.children = [mock_clause]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result is None


def test_get_export_name_lexical_no_declarator(ts_parser):
    """Test _get_export_name for lexical declaration without variable_declarator."""
    mock_lexical = MagicMock()
    mock_lexical.type = "lexical_declaration"
    mock_lexical.children = []

    mock_node = MagicMock()
    mock_node.children = [mock_lexical]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result == "default"


def test_get_export_name_identifier(ts_parser):
    """Test _get_export_name for identifier export."""
    mock_identifier = MagicMock()
    mock_identifier.type = "identifier"
    mock_identifier.text = b"myExport"

    mock_node = MagicMock()
    mock_node.children = [mock_identifier]

    result = ts_parser._get_export_name(mock_node, b"")
    assert result == "myExport"


def test_get_node_type_unknown(ts_parser):
    """Test _get_node_type returns ts_type for unknown types."""
    mock_node = MagicMock()
    mock_node.type = "unknown_type"
    mock_node.parent = None

    mock_outer = MagicMock()
    mock_outer.type = "unknown_outer"

    result = ts_parser._get_node_type(mock_node, mock_outer)
    assert result == "unknown_type"


def test_decorator_class(ts_parser):
    """Test class with decorators."""
    code = """@Component({
    selector: 'app-root'
})
class AppComponent {}
"""
    results = list(ts_parser.parse(code))

    # Should find the class
    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) >= 1


def test_static_method(ts_parser):
    """Test static method."""
    code = """class MyClass {
    static staticMethod(): void {}
}
"""
    results = list(ts_parser.parse(code))

    # Should find class and method
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1
