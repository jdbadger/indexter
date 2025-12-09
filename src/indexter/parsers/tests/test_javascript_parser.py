"""Tests for the JavaScriptParser."""

from unittest.mock import MagicMock, Mock

import pytest
from tree_sitter import Node

from indexter.parsers.javascript import JavaScriptParser


@pytest.fixture
def js_parser():
    """Create a JavaScriptParser instance for testing."""
    return JavaScriptParser()


@pytest.fixture
def simple_function_js():
    """Sample JavaScript with a simple function."""
    return """function greet(name) {
    return `Hello, ${name}!`;
}"""


@pytest.fixture
def arrow_function_js():
    """Sample JavaScript with arrow functions."""
    return """const add = (a, b) => a + b;

const multiply = (x, y) => {
    return x * y;
};"""


@pytest.fixture
def class_js():
    """Sample JavaScript with a class."""
    return """class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
}"""


@pytest.fixture
def async_function_js():
    """Sample JavaScript with async functions."""
    return """async function fetchData() {
    const response = await fetch('/api/data');
    return response.json();
}"""


@pytest.fixture
def generator_function_js():
    """Sample JavaScript with generator function."""
    return """function* countUp(max) {
    for (let i = 0; i < max; i++) {
        yield i;
    }
}"""


@pytest.fixture
def jsdoc_function_js():
    """Sample JavaScript with JSDoc comments."""
    return """/**
 * Calculate the sum of two numbers
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}"""


@pytest.fixture
def import_export_js():
    """Sample JavaScript with imports and exports."""
    return """import { useState } from 'react';
import axios from 'axios';

export function Component() {
    return null;
}

export default MyComponent;"""


@pytest.fixture
def constants_js():
    """Sample JavaScript with constants."""
    return """const API_KEY = 'secret123';
const MAX_RETRIES = 3;
const config = { timeout: 5000 };"""


def test_parser_initialization(js_parser):
    """Test that JavaScriptParser initializes correctly."""
    assert js_parser.language == "javascript"
    assert js_parser.tslanguage is not None
    assert js_parser.tsparser is not None


def test_query_str(js_parser):
    """Test that query_str returns a valid query string."""
    query = js_parser.query_str
    assert "function_declaration" in query
    assert "arrow_function" in query
    assert "class_declaration" in query
    assert "method_definition" in query
    assert "import_statement" in query
    assert "export_statement" in query


def test_parse_simple_function(js_parser, simple_function_js):
    """Test parsing a simple function declaration."""
    results = list(js_parser.parse(simple_function_js))

    assert len(results) == 1
    content, node_info = results[0]

    assert "function greet" in content
    assert "name" in content

    assert node_info["language"] == "javascript"
    assert node_info["node_type"] == "function"
    assert node_info["node_name"] == "greet"
    assert node_info["start_line"] == 1
    assert node_info["parent_scope"] is None
    assert node_info["signature"] is not None
    assert "(name)" in node_info["signature"]


def test_parse_arrow_function(js_parser, arrow_function_js):
    """Test parsing arrow functions."""
    results = list(js_parser.parse(arrow_function_js))

    assert len(results) == 2

    # First arrow function
    content1, node_info1 = results[0]
    assert node_info1["node_name"] == "add"
    assert node_info1["node_type"] == "function"
    assert node_info1["extra"]["is_arrow"] == "true"

    # Second arrow function
    content2, node_info2 = results[1]
    assert node_info2["node_name"] == "multiply"
    assert node_info2["node_type"] == "function"
    assert node_info2["extra"]["is_arrow"] == "true"


def test_parse_class(js_parser, class_js):
    """Test parsing a class with methods."""
    results = list(js_parser.parse(class_js))

    # Should find class and two methods
    assert len(results) >= 3

    # Check class
    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 1
    assert class_results[0][1]["node_name"] == "Calculator"

    # Check methods
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) == 2

    method_names = [r[1]["node_name"] for r in method_results]
    assert "add" in method_names
    assert "subtract" in method_names

    # Methods should have parent_scope set to class name
    for _, info in method_results:
        assert info["parent_scope"] == "Calculator"


def test_parse_async_function(js_parser, async_function_js):
    """Test parsing async functions."""
    results = list(js_parser.parse(async_function_js))

    assert len(results) == 1
    content, node_info = results[0]

    assert node_info["node_name"] == "fetchData"
    assert node_info["extra"]["is_async"] == "true"
    assert "async" in content


def test_parse_generator_function(js_parser, generator_function_js):
    """Test parsing generator functions."""
    results = list(js_parser.parse(generator_function_js))

    assert len(results) == 1
    content, node_info = results[0]

    assert node_info["node_name"] == "countUp"
    assert node_info["extra"]["is_generator"] == "true"
    assert "yield" in content


def test_parse_jsdoc(js_parser, jsdoc_function_js):
    """Test parsing JSDoc comments."""
    results = list(js_parser.parse(jsdoc_function_js))

    assert len(results) == 1
    content, node_info = results[0]

    assert node_info["documentation"] is not None
    assert "Calculate the sum" in node_info["documentation"]
    assert "number" in node_info["documentation"]


def test_parse_imports(js_parser, import_export_js):
    """Test parsing import statements."""
    results = list(js_parser.parse(import_export_js))

    import_results = [r for r in results if r[1]["node_type"] == "import"]
    assert len(import_results) >= 1

    # Check import sources
    sources = [r[1]["node_name"] for r in import_results]
    assert "react" in sources or any("react" in s for s in sources)


def test_parse_exports(js_parser, import_export_js):
    """Test parsing export statements."""
    results = list(js_parser.parse(import_export_js))

    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 1


def test_parse_constants(js_parser, constants_js):
    """Test parsing constant declarations."""
    results = list(js_parser.parse(constants_js))

    # Should find only UPPER_CASE constants
    const_results = [r for r in results if r[1]["node_type"] == "constant"]

    const_names = [r[1]["node_name"] for r in const_results]
    assert "API_KEY" in const_names
    assert "MAX_RETRIES" in const_names
    # 'config' should not be included (not UPPER_CASE)
    assert "config" not in const_names


def test_get_node_type_function(js_parser):
    """Test _get_node_type for function declarations."""
    mock_node = Mock(spec=Node)
    mock_node.type = "function_declaration"
    mock_node.parent = None

    node_type = js_parser._get_node_type(mock_node, mock_node)
    assert node_type == "function"


def test_get_node_type_class(js_parser):
    """Test _get_node_type for class declarations."""
    mock_node = Mock(spec=Node)
    mock_node.type = "class_declaration"

    node_type = js_parser._get_node_type(mock_node, mock_node)
    assert node_type == "class"


def test_get_node_type_method(js_parser):
    """Test _get_node_type for method definitions."""
    mock_node = Mock(spec=Node)
    mock_node.type = "method_definition"

    node_type = js_parser._get_node_type(mock_node, mock_node)
    assert node_type == "method"


def test_parse_jsdoc_single_line():
    """Test JSDoc parsing with single-line comment."""
    parser = JavaScriptParser()
    comment = "/** Single line comment */"

    result = parser._parse_jsdoc(comment)
    assert result == "Single line comment"


def test_parse_jsdoc_multi_line():
    """Test JSDoc parsing with multi-line comment."""
    parser = JavaScriptParser()
    comment = """/**
 * First line
 * Second line
 * Third line
 */"""

    result = parser._parse_jsdoc(comment)
    assert "First line" in result
    assert "Second line" in result
    assert "Third line" in result


def test_parse_jsdoc_not_jsdoc():
    """Test JSDoc parsing with regular comment."""
    parser = JavaScriptParser()
    comment = "// Not a JSDoc comment"

    result = parser._parse_jsdoc(comment)
    assert result is None


def test_parse_jsdoc_block_comment():
    """Test JSDoc parsing with block comment (not JSDoc)."""
    parser = JavaScriptParser()
    comment = "/* Not a JSDoc comment */"

    result = parser._parse_jsdoc(comment)
    assert result is None


def test_is_async_function(js_parser):
    """Test _is_async for async functions."""
    mock_async = Mock(spec=Node)
    mock_async.type = "async"

    mock_node = Mock(spec=Node)
    mock_node.type = "function_declaration"
    mock_node.children = [mock_async]

    assert js_parser._is_async(mock_node) is True


def test_is_not_async_function(js_parser):
    """Test _is_async for non-async functions."""
    mock_node = Mock(spec=Node)
    mock_node.type = "function_declaration"
    mock_node.children = []

    assert js_parser._is_async(mock_node) is False


def test_is_generator_function(js_parser):
    """Test _is_generator for generator functions."""
    mock_node = Mock(spec=Node)
    mock_node.type = "generator_function_declaration"

    assert js_parser._is_generator(mock_node) is True


def test_is_not_generator_function(js_parser):
    """Test _is_generator for non-generator functions."""
    mock_node = Mock(spec=Node)
    mock_node.type = "function_declaration"
    mock_node.children = []

    assert js_parser._is_generator(mock_node) is False


def test_is_const_declaration(js_parser):
    """Test _is_const_declaration."""
    mock_const = Mock(spec=Node)
    mock_const.type = "const"

    mock_node = Mock(spec=Node)
    mock_node.children = [mock_const]

    assert js_parser._is_const_declaration(mock_node) is True


def test_is_not_const_declaration(js_parser):
    """Test _is_const_declaration with let/var."""
    mock_let = Mock(spec=Node)
    mock_let.type = "let"

    mock_node = Mock(spec=Node)
    mock_node.children = [mock_let]

    assert js_parser._is_const_declaration(mock_node) is False


def test_is_constant_uppercase(js_parser):
    """Test _is_constant with UPPER_CASE names."""
    assert js_parser._is_constant("API_KEY") is True
    assert js_parser._is_constant("MAX_RETRIES") is True
    assert js_parser._is_constant("CONFIG") is True


def test_is_not_constant_camelcase(js_parser):
    """Test _is_constant with camelCase names."""
    assert js_parser._is_constant("apiKey") is False
    assert js_parser._is_constant("maxRetries") is False
    assert js_parser._is_constant("config") is False


def test_parse_empty_js(js_parser):
    """Test parsing empty JavaScript."""
    results = list(js_parser.parse(""))
    assert len(results) == 0


def test_parse_comments_only(js_parser):
    """Test parsing JavaScript with only comments."""
    js = """// Comment 1
/* Comment 2 */
/** JSDoc comment */"""

    results = list(js_parser.parse(js))
    assert len(results) == 0


def test_parse_variable_declarations(js_parser):
    """Test that non-constant variable declarations are skipped."""
    js = """let counter = 0;
var oldStyle = 'test';
const regular = 'value';"""

    results = list(js_parser.parse(js))

    # Should not include let/var or lowercase const
    const_results = [r for r in results if r[1]["node_type"] == "constant"]
    assert len(const_results) == 0


def test_parse_method_in_class(js_parser):
    """Test that methods have correct parent scope."""
    js = """class MyClass {
    myMethod() {
        return 'test';
    }
}"""

    results = list(js_parser.parse(js))

    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1

    for _, info in method_results:
        assert info["parent_scope"] == "MyClass"


def test_parse_nested_function(js_parser):
    """Test parsing nested functions."""
    js = """function outer() {
    function inner() {
        return 'nested';
    }
    return inner;
}"""

    results = list(js_parser.parse(js))

    # Should find both outer and inner functions
    function_names = [r[1]["node_name"] for r in results]
    assert "outer" in function_names
    assert "inner" in function_names


def test_byte_positions_accuracy(js_parser):
    """Test that byte positions are accurate."""
    js = """function first() {}
function second() {}"""

    results = list(js_parser.parse(js))

    assert len(results) == 2

    # Check that extracting content by byte positions works
    source_bytes = js.encode()
    for content, node_info in results:
        extracted = source_bytes[node_info["start_byte"] : node_info["end_byte"]].decode()
        assert extracted == content


def test_line_numbers_accuracy(js_parser):
    """Test that line numbers are accurate (1-based)."""
    js = """function first() {
    return 1;
}

function second() {
    return 2;
}"""

    results = list(js_parser.parse(js))

    assert len(results) == 2

    first_info = results[0][1]
    assert first_info["start_line"] == 1

    second_info = results[1][1]
    assert second_info["start_line"] == 5


def test_signature_extraction(js_parser):
    """Test signature extraction for functions."""
    js = """function add(a, b, c) {
    return a + b + c;
}"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["signature"] is not None
    assert "add" in node_info["signature"]
    assert "(a, b, c)" in node_info["signature"]


def test_arrow_function_signature(js_parser):
    """Test signature extraction for arrow functions."""
    js = """const square = (x) => x * x;"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["signature"] is not None
    assert "=>" in node_info["signature"]


def test_extra_metadata(js_parser):
    """Test extra metadata fields."""
    js = """async function* asyncGen() {
    yield 1;
}"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    assert "is_async" in node_info["extra"]
    assert "is_generator" in node_info["extra"]
    assert "is_arrow" in node_info["extra"]


def test_parse_export_function(js_parser):
    """Test parsing exported functions."""
    js = """export function exportedFunc() {
    return 'exported';
}"""

    results = list(js_parser.parse(js))

    # Should find the export statement
    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 1


def test_parse_export_default(js_parser):
    """Test parsing default exports."""
    js = """export default function() {
    return 'default';
}"""

    results = list(js_parser.parse(js))

    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 1


def test_parse_named_exports(js_parser):
    """Test parsing named exports."""
    js = """const foo = 1;
const bar = 2;
export { foo, bar };"""

    results = list(js_parser.parse(js))

    # Should find export statement
    export_results = [r for r in results if r[1]["node_type"] == "export"]
    assert len(export_results) >= 1


def test_parse_class_with_constructor(js_parser):
    """Test parsing class with constructor."""
    js = """class Person {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
}"""

    results = list(js_parser.parse(js))

    # Should find class and methods (including constructor)
    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 1

    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1


def test_parse_async_arrow_function(js_parser):
    """Test parsing async arrow functions."""
    js = """const fetchUser = async (id) => {
    const response = await fetch(`/user/${id}`);
    return response.json();
};"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["is_async"] == "true"
    assert node_info["extra"]["is_arrow"] == "true"


def test_documentation_is_none_without_jsdoc(js_parser):
    """Test that functions without JSDoc have None documentation."""
    js = """function noDoc() {
    return 'test';
}"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["documentation"] is None


def test_parse_multiple_classes(js_parser):
    """Test parsing multiple classes."""
    js = """class First {
    method1() {}
}

class Second {
    method2() {}
}"""

    results = list(js_parser.parse(js))

    class_results = [r for r in results if r[1]["node_type"] == "class"]
    assert len(class_results) == 2

    class_names = [r[1]["node_name"] for r in class_results]
    assert "First" in class_names
    assert "Second" in class_names


def test_parser_language_property(js_parser):
    """Test that parser reports correct language."""
    js = """function test() {}"""
    results = list(js_parser.parse(js))

    assert len(results) == 1
    _, node_info = results[0]
    assert node_info["language"] == "javascript"


def test_process_match_no_def_nodes(js_parser):
    """Test process_match returns None when no def nodes."""
    source_bytes = b"const x = 1;"
    match = {}

    result = js_parser.process_match(match, source_bytes)
    assert result is None


def test_process_match_lexical_in_export(js_parser):
    """Test that lexical declarations in exports are skipped."""
    source_bytes = b"export const X = 1;"

    mock_export = Mock(spec=Node)
    mock_export.type = "export_statement"

    mock_lexical = Mock(spec=Node)
    mock_lexical.type = "lexical_declaration"
    mock_lexical.parent = mock_export

    match = {"def": [mock_lexical]}

    result = js_parser.process_match(match, source_bytes)
    # Should be skipped because it's inside an export statement
    assert result is None


def test_get_parent_scope_nested_class(js_parser):
    """Test _get_parent_scope for deeply nested nodes."""
    mock_name = Mock(spec=Node)
    mock_name.text = b"MyClass"

    mock_class = Mock(spec=Node)
    mock_class.type = "class_declaration"
    mock_class.child_by_field_name = Mock(return_value=mock_name)

    mock_class_body = Mock(spec=Node)
    mock_class_body.type = "class_body"
    mock_class_body.parent = mock_class

    mock_method = Mock(spec=Node)
    mock_method.type = "method_definition"
    mock_method.parent = mock_class_body

    parent_scope = js_parser._get_parent_scope(mock_method)
    assert parent_scope == "MyClass"


def test_get_parent_scope_no_class(js_parser):
    """Test _get_parent_scope returns None for top-level functions."""
    mock_node = Mock(spec=Node)
    mock_node.parent = None

    parent_scope = js_parser._get_parent_scope(mock_node)
    assert parent_scope is None


def test_get_export_name_function(js_parser):
    """Test _get_export_name for exported function."""
    mock_name = Mock(spec=Node)
    mock_name.text = b"myFunc"

    mock_func = Mock(spec=Node)
    mock_func.type = "function_declaration"
    mock_func.child_by_field_name = Mock(return_value=mock_name)

    mock_export = Mock(spec=Node)
    mock_export.children = [mock_func]

    name = js_parser._get_export_name(mock_export, b"")
    assert name == "myFunc"


def test_get_export_name_default(js_parser):
    """Test _get_export_name for default export."""
    mock_export = Mock(spec=Node)
    mock_export.children = []

    name = js_parser._get_export_name(mock_export, b"")
    assert name == "default"


def test_get_signature_arrow_with_body(js_parser):
    """Test signature extraction for arrow function with body."""
    js = """const func = (a, b) => {
    return a + b;
};"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    # Signature should include params and =>
    assert node_info["signature"] is not None


def test_get_content(js_parser):
    """Test _get_content extracts correct source text."""
    source = b"function test() { return 1; }"

    mock_node = Mock(spec=Node)
    mock_node.start_byte = 0
    mock_node.end_byte = len(source)

    content = js_parser._get_content(mock_node, source)
    assert content == source.decode()


def test_parse_import_from(js_parser):
    """Test parsing import from statements."""
    js = """import { Component } from 'react';"""

    results = list(js_parser.parse(js))

    import_results = [r for r in results if r[1]["node_type"] == "import"]
    assert len(import_results) >= 1


def test_parse_import_default(js_parser):
    """Test parsing default import statements."""
    js = """import React from 'react';"""

    results = list(js_parser.parse(js))

    import_results = [r for r in results if r[1]["node_type"] == "import"]
    assert len(import_results) >= 1


def test_extra_metadata_regular_function(js_parser):
    """Test extra metadata for regular functions."""
    js = """function regular() {
    return 'test';
}"""

    results = list(js_parser.parse(js))
    assert len(results) == 1

    content, node_info = results[0]
    assert node_info["extra"]["is_async"] == "false"
    assert node_info["extra"]["is_generator"] == "false"
    assert node_info["extra"]["is_arrow"] == "false"


def test_parse_method_with_async(js_parser):
    """Test parsing async methods in classes."""
    js = """class API {
    async getData() {
        return await fetch('/data');
    }
}"""

    results = list(js_parser.parse(js))

    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1

    for _, info in method_results:
        if info["node_name"] == "getData":
            assert info["extra"]["is_async"] == "true"


def test_jsdoc_with_empty_lines(js_parser):
    """Test JSDoc parsing with empty lines."""
    comment = """/**
 * First paragraph
 * 
 * Second paragraph
 */"""

    result = js_parser._parse_jsdoc(comment)
    assert result is not None
    assert "First paragraph" in result
    assert "Second paragraph" in result


def test_parse_export_anonymous_class(js_parser):
    """Test export of anonymous class (export default)."""
    code = """
    export default class {
        method() {}
    }
    """
    elements = list(js_parser.parse(code))
    exports = [e for e in elements if e[1]["node_type"] == "export"]
    assert len(exports) == 1
    assert exports[0][1]["node_name"] == "default"


def test_parse_export_lexical_declaration(js_parser):
    """Test export of lexical declaration (const/let)."""
    code = """
    export const API_KEY = "secret";
    """
    elements = list(js_parser.parse(code))
    exports = [e for e in elements if e[1]["node_type"] == "export"]
    assert len(exports) == 1
    assert exports[0][1]["node_name"] == "API_KEY"


def test_parse_export_identifier(js_parser):
    """Test export of identifier."""
    code = """
    const value = 42;
    export default value;
    """
    elements = list(js_parser.parse(code))
    # Should find the export
    exports = [e for e in elements if e[1]["node_type"] == "export"]
    assert len(exports) == 1
    assert exports[0][1]["node_name"] == "value"


def test_process_match_no_name_or_source(js_parser):
    """Test process_match when there's no name or source (edge case)."""
    # Empty match dictionary (no name or source)
    match = {}
    source_bytes = b"some code"

    result = js_parser.process_match(match, source_bytes)
    assert result is None


def test_get_documentation_no_parent(js_parser):
    """Test _get_documentation when node has no parent."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = js_parser._get_documentation(mock_node, b"code")
    assert result is None


def test_get_signature_arrow_without_body(js_parser):
    """Test _get_signature for arrow function with params but no body field."""
    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.start_byte = 0
    mock_node.end_byte = 10

    # No body field returns None
    mock_node.child_by_field_name = MagicMock(return_value=None)
    mock_node.children = []

    source = b"x => x * 2"
    result = js_parser._get_signature(mock_node, source)
    assert result is None


def test_get_signature_function_without_body(js_parser):
    """Test _get_signature for function without body field."""
    mock_node = MagicMock()
    mock_node.type = "function_declaration"
    mock_node.start_byte = 0
    mock_node.end_byte = 20
    mock_node.child_by_field_name = MagicMock(return_value=None)

    source = b"function test()"
    result = js_parser._get_signature(mock_node, source)
    assert result is None


def test_is_generator_function_with_asterisk(js_parser):
    """Test _is_generator for function with asterisk child."""
    mock_node = MagicMock()
    mock_node.type = "function_declaration"

    asterisk = MagicMock()
    asterisk.type = "*"

    mock_node.children = [asterisk]

    result = js_parser._is_generator(mock_node)
    assert result is True


def test_get_export_name_class_without_name(js_parser):
    """Test _get_export_name for class without name field."""
    mock_node = MagicMock()
    mock_node.type = "export_statement"

    class_node = MagicMock()
    class_node.type = "class_declaration"
    class_node.child_by_field_name = MagicMock(return_value=None)  # No name

    mock_node.children = [class_node]

    result = js_parser._get_export_name(mock_node, b"export default class {}")
    assert result == "default"


def test_get_export_name_function_without_name(js_parser):
    """Test _get_export_name for function without name field."""
    mock_node = MagicMock()
    mock_node.type = "export_statement"

    func_node = MagicMock()
    func_node.type = "function_declaration"
    func_node.child_by_field_name = MagicMock(return_value=None)  # No name

    mock_node.children = [func_node]

    result = js_parser._get_export_name(mock_node, b"export default function() {}")
    assert result == "default"


def test_get_export_name_lexical_without_name(js_parser):
    """Test _get_export_name for lexical declaration without name."""
    mock_node = MagicMock()
    mock_node.type = "export_statement"

    lexical_node = MagicMock()
    lexical_node.type = "lexical_declaration"

    var_decl = MagicMock()
    var_decl.type = "variable_declarator"
    var_decl.child_by_field_name = MagicMock(return_value=None)  # No name

    lexical_node.children = [var_decl]
    mock_node.children = [lexical_node]

    result = js_parser._get_export_name(mock_node, b"export const;")
    assert result is None


def test_get_node_type_returns_ts_type(js_parser):
    """Test _get_node_type returns ts_type when no special case matches."""
    mock_node = MagicMock()
    mock_node.type = "variable_declaration"
    mock_node.parent = None  # No parent scope

    # Create outer_node that's not import/export/lexical_declaration
    mock_outer = MagicMock()
    mock_outer.type = "program"

    result = js_parser._get_node_type(mock_node, mock_outer)
    assert result == "variable_declaration"


def test_get_signature_arrow_with_body_field(js_parser):
    """Test _get_signature for arrow function with body field (no arrow child)."""
    mock_node = MagicMock()
    mock_node.type = "arrow_function"
    mock_node.start_byte = 0

    # Mock body field
    body_node = MagicMock()
    body_node.start_byte = 6

    def child_by_field(name):
        if name == "body":
            return body_node
        return None

    mock_node.child_by_field_name = child_by_field
    mock_node.children = []  # No => child

    source = b"x => { return x * 2; }"
    result = js_parser._get_signature(mock_node, source)
    assert result == "x => {"
