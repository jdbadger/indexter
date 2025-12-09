"""Tests for the RustParser."""

from unittest.mock import MagicMock

import pytest

from indexter.parsers.rust import RustParser


@pytest.fixture
def rust_parser():
    """Create a RustParser instance for testing."""
    return RustParser()


@pytest.fixture
def simple_function():
    """Sample Rust with a simple function."""
    return """fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}
"""


@pytest.fixture
def simple_struct():
    """Sample Rust with a simple struct."""
    return """/// A person struct
struct Person {
    name: String,
    age: u32,
}
"""


@pytest.fixture
def simple_enum():
    """Sample Rust with a simple enum."""
    return """/// An option-like enum
enum MyOption {
    Some(i32),
    None,
}
"""


@pytest.fixture
def simple_trait():
    """Sample Rust with a simple trait."""
    return """/// A greeting trait
trait Greet {
    fn greet(&self) -> String;
}
"""


@pytest.fixture
def impl_block():
    """Sample Rust with an impl block."""
    return """impl Person {
    fn new(name: String, age: u32) -> Self {
        Person { name, age }
    }
}
"""


@pytest.fixture
def async_function():
    """Sample Rust with an async function."""
    return """async fn fetch_data(url: &str) -> Result<String, Error> {
    // implementation
}
"""


@pytest.fixture
def unsafe_function():
    """Sample Rust with an unsafe function."""
    return """unsafe fn raw_ptr_deref(ptr: *const i32) -> i32 {
    *ptr
}
"""


@pytest.fixture
def pub_function():
    """Sample Rust with a public function."""
    return """pub fn public_api() -> i32 {
    42
}
"""


@pytest.fixture
def constant():
    """Sample Rust with a constant."""
    return """const MAX_SIZE: usize = 100;
"""


@pytest.fixture
def static_var():
    """Sample Rust with a static variable."""
    return """static GLOBAL_COUNT: AtomicUsize = AtomicUsize::new(0);
"""


@pytest.fixture
def type_alias():
    """Sample Rust with a type alias."""
    return """type Result<T> = std::result::Result<T, Error>;
"""


@pytest.fixture
def use_declaration():
    """Sample Rust with use declarations."""
    return """use std::io::Read;
use std::collections::HashMap as HMap;
"""


@pytest.fixture
def module_decl():
    """Sample Rust with a module declaration."""
    return """mod utils {
    pub fn helper() {}
}
"""


def test_parser_initialization(rust_parser):
    """Test that RustParser initializes correctly."""
    assert rust_parser.language == "rust"
    assert rust_parser.tslanguage is not None
    assert rust_parser.tsparser is not None


def test_query_str(rust_parser):
    """Test that query_str returns a valid query string."""
    query = rust_parser.query_str
    assert "function_item" in query
    assert "struct_item" in query
    assert "enum_item" in query
    assert "trait_item" in query
    assert "impl_item" in query
    assert "const_item" in query
    assert "static_item" in query
    assert "type_item" in query
    assert "use_declaration" in query
    assert "mod_item" in query


def test_parse_simple_function(rust_parser, simple_function):
    """Test parsing a simple function."""
    results = list(rust_parser.parse(simple_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "function"
    assert info["node_name"] == "greet"
    assert info["language"] == "rust"
    assert "fn greet(name: &str) -> String" in info["signature"]
    assert info["parent_scope"] is None
    assert info["extra"]["is_async"] == "false"
    assert info["extra"]["is_unsafe"] == "false"


def test_parse_simple_struct(rust_parser, simple_struct):
    """Test parsing a simple struct."""
    results = list(rust_parser.parse(simple_struct))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "struct"
    assert info["node_name"] == "Person"
    assert info["documentation"] == "A person struct"


def test_parse_simple_enum(rust_parser, simple_enum):
    """Test parsing a simple enum."""
    results = list(rust_parser.parse(simple_enum))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "enum"
    assert info["node_name"] == "MyOption"
    assert info["documentation"] == "An option-like enum"


def test_parse_simple_trait(rust_parser, simple_trait):
    """Test parsing a simple trait."""
    results = list(rust_parser.parse(simple_trait))

    assert len(results) >= 1

    # Find the trait
    trait_result = [r for r in results if r[1]["node_type"] == "trait"][0]
    assert trait_result[1]["node_name"] == "Greet"
    assert trait_result[1]["documentation"] == "A greeting trait"


def test_parse_impl_block(rust_parser, impl_block):
    """Test parsing an impl block."""
    results = list(rust_parser.parse(impl_block))

    # Should find: impl block + method
    assert len(results) >= 2

    # Check impl
    impl_result = [r for r in results if r[1]["node_type"] == "impl"][0]
    assert impl_result[1]["node_name"] == "Person"

    # Check method
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) == 1
    assert method_results[0][1]["node_name"] == "new"
    assert method_results[0][1]["parent_scope"] == "Person"


def test_parse_async_function(rust_parser, async_function):
    """Test parsing an async function."""
    results = list(rust_parser.parse(async_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "fetch_data"
    assert info["extra"]["is_async"] == "true"
    assert "async fn" in info["signature"]


def test_parse_unsafe_function(rust_parser, unsafe_function):
    """Test parsing an unsafe function."""
    results = list(rust_parser.parse(unsafe_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "raw_ptr_deref"
    assert info["extra"]["is_unsafe"] == "true"
    assert "unsafe fn" in info["signature"]


def test_parse_pub_function(rust_parser, pub_function):
    """Test parsing a public function."""
    results = list(rust_parser.parse(pub_function))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_name"] == "public_api"
    assert info["extra"]["is_pub"] == "true"
    assert "pub fn" in info["signature"]


def test_parse_constant(rust_parser, constant):
    """Test parsing a constant."""
    results = list(rust_parser.parse(constant))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "constant"
    assert info["node_name"] == "MAX_SIZE"


def test_parse_static_var(rust_parser, static_var):
    """Test parsing a static variable."""
    results = list(rust_parser.parse(static_var))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "static"
    assert info["node_name"] == "GLOBAL_COUNT"


def test_parse_type_alias(rust_parser, type_alias):
    """Test parsing a type alias."""
    results = list(rust_parser.parse(type_alias))

    assert len(results) == 1
    content, info = results[0]

    assert info["node_type"] == "type_alias"
    assert info["node_name"] == "Result"


def test_parse_use_declarations(rust_parser, use_declaration):
    """Test parsing use declarations."""
    results = list(rust_parser.parse(use_declaration))

    assert len(results) == 2

    # First import
    assert results[0][1]["node_type"] == "import"
    assert results[0][1]["node_name"] == "Read"

    # Second import with alias
    assert results[1][1]["node_type"] == "import"
    assert results[1][1]["node_name"] == "HMap"


def test_parse_module(rust_parser, module_decl):
    """Test parsing a module declaration."""
    results = list(rust_parser.parse(module_decl))

    # Should find: module + function inside
    mod_results = [r for r in results if r[1]["node_type"] == "module"]
    assert len(mod_results) == 1
    assert mod_results[0][1]["node_name"] == "utils"


def test_parse_empty_rust(rust_parser):
    """Test parsing empty Rust code."""
    results = list(rust_parser.parse(""))
    assert len(results) == 0


def test_parse_comments_only(rust_parser):
    """Test parsing Rust with only comments."""
    code = """// This is a comment
// Another comment
"""
    results = list(rust_parser.parse(code))
    assert len(results) == 0


def test_doc_comment_triple_slash(rust_parser):
    """Test triple slash doc comments."""
    code = """/// This is a doc comment
/// on multiple lines
fn documented() {}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    doc = results[0][1]["documentation"]
    assert "This is a doc comment" in doc
    assert "on multiple lines" in doc


def test_doc_comment_block(rust_parser):
    """Test block doc comments."""
    code = """/**
 * This is a block doc comment
 * with multiple lines
 */
fn documented() {}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    doc = results[0][1]["documentation"]
    assert "This is a block doc comment" in doc
    assert "with multiple lines" in doc


def test_inner_doc_comment(rust_parser):
    """Test inner doc comments (//!)."""
    code = """//! Module level documentation
//! More details
fn func() {}
"""
    results = list(rust_parser.parse(code))

    # The function should be found
    func_results = [r for r in results if r[1]["node_name"] == "func"]
    assert len(func_results) == 1


def test_attributes(rust_parser):
    """Test parsing attributes."""
    code = """#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Attributed {
    field: i32,
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    attributes = results[0][1]["extra"]["attributes"]
    assert "#[derive(Debug, Clone)]" in attributes
    assert "#[allow(dead_code)]" in attributes


def test_trait_method_signature(rust_parser):
    """Test trait method signature extraction."""
    code = """trait Display {
    fn display(&self) -> String;
}
"""
    results = list(rust_parser.parse(code))

    # Should find trait and method
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    if method_results:
        assert "fn display(&self) -> String" in method_results[0][1]["signature"]


def test_method_in_trait_impl(rust_parser):
    """Test method in trait impl."""
    code = """impl Greet for Person {
    fn greet(&self) -> String {
        format!("Hi, {}", self.name)
    }
}
"""
    results = list(rust_parser.parse(code))

    # Should find impl and method
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1
    assert method_results[0][1]["node_name"] == "greet"
    assert method_results[0][1]["parent_scope"] == "Person"


def test_async_unsafe_function(rust_parser):
    """Test function that is both async and unsafe."""
    code = """async unsafe fn dangerous_async() {
    // implementation
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    info = results[0][1]
    assert info["extra"]["is_async"] == "true"
    assert info["extra"]["is_unsafe"] == "true"


def test_pub_async_function(rust_parser):
    """Test public async function."""
    code = """pub async fn public_async() -> Result<()> {
    Ok(())
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    info = results[0][1]
    assert info["extra"]["is_async"] == "true"
    assert info["extra"]["is_pub"] == "true"


def test_generic_function(rust_parser):
    """Test parsing generic function."""
    code = """fn generic<T>(value: T) -> T {
    value
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "generic"
    assert "T" in results[0][1]["signature"]


def test_generic_struct(rust_parser):
    """Test parsing generic struct."""
    code = """struct Container<T> {
    value: T,
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "Container"


def test_lifetime_annotations(rust_parser):
    """Test parsing with lifetime annotations."""
    code = """fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "longest"
    assert "'a" in results[0][1]["signature"]


def test_where_clause(rust_parser):
    """Test parsing function with where clause."""
    code = """fn process<T>(item: T)
where
    T: Clone + Debug,
{
    // implementation
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_name"] == "process"


def test_multiple_structs(rust_parser):
    """Test parsing multiple structs."""
    code = """struct First {
    a: i32,
}

struct Second {
    b: String,
}

struct Third;
"""
    results = list(rust_parser.parse(code))

    struct_results = [r for r in results if r[1]["node_type"] == "struct"]
    assert len(struct_results) == 3

    struct_names = [r[1]["node_name"] for r in struct_results]
    assert "First" in struct_names
    assert "Second" in struct_names
    assert "Third" in struct_names


def test_tuple_struct(rust_parser):
    """Test parsing tuple struct."""
    code = """struct Point(i32, i32);
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_type"] == "struct"
    assert results[0][1]["node_name"] == "Point"


def test_enum_with_variants(rust_parser):
    """Test parsing enum with various variant types."""
    code = """enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
"""
    results = list(rust_parser.parse(code))

    assert len(results) == 1
    assert results[0][1]["node_type"] == "enum"
    assert results[0][1]["node_name"] == "Message"


def test_unsafe_impl(rust_parser):
    """Test parsing unsafe impl."""
    code = """unsafe impl Send for MyType {}
"""
    results = list(rust_parser.parse(code))

    impl_results = [r for r in results if r[1]["node_type"] == "impl"]
    assert len(impl_results) == 1
    assert impl_results[0][1]["extra"]["is_unsafe"] == "true"


def test_unsafe_trait(rust_parser):
    """Test parsing unsafe trait."""
    code = """unsafe trait UnsafeTrait {
    fn unsafe_method(&self);
}
"""
    results = list(rust_parser.parse(code))

    trait_results = [r for r in results if r[1]["node_type"] == "trait"]
    assert len(trait_results) == 1
    assert trait_results[0][1]["extra"]["is_unsafe"] == "true"


def test_byte_positions(rust_parser):
    """Test byte position tracking."""
    code = """fn first() {}

fn second() {}
"""
    results = list(rust_parser.parse(code))

    for _, info in results:
        assert info["start_byte"] >= 0
        assert info["end_byte"] > info["start_byte"]


def test_line_numbers(rust_parser):
    """Test line number tracking (1-based)."""
    code = """fn func1() {}

fn func2() {}
"""
    results = list(rust_parser.parse(code))

    assert results[0][1]["start_line"] == 1
    assert results[1][1]["start_line"] == 3


def test_get_content(rust_parser):
    """Test _get_content extracts source correctly."""
    code = b"fn test() {}"
    mock_node = MagicMock()
    mock_node.start_byte = 0
    mock_node.end_byte = len(code)

    result = rust_parser._get_content(mock_node, code)
    assert result == code.decode()


def test_get_node_type_function(rust_parser):
    """Test _get_node_type for functions."""
    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.parent = MagicMock()
    mock_node.parent.type = "source_file"

    # Mock _get_parent_scope to return None (top-level function)
    original_get_parent_scope = rust_parser._get_parent_scope
    rust_parser._get_parent_scope = lambda n: None

    result = rust_parser._get_node_type(mock_node)
    assert result == "function"

    rust_parser._get_parent_scope = original_get_parent_scope


def test_get_node_type_method(rust_parser):
    """Test _get_node_type for methods."""
    mock_node = MagicMock()
    mock_node.type = "function_item"

    # Mock _get_parent_scope to return a class name
    original_get_parent_scope = rust_parser._get_parent_scope
    rust_parser._get_parent_scope = lambda n: "MyStruct"

    result = rust_parser._get_node_type(mock_node)
    assert result == "method"

    rust_parser._get_parent_scope = original_get_parent_scope


def test_get_node_type_struct(rust_parser):
    """Test _get_node_type for structs."""
    mock_node = MagicMock()
    mock_node.type = "struct_item"

    result = rust_parser._get_node_type(mock_node)
    assert result == "struct"


def test_get_node_type_enum(rust_parser):
    """Test _get_node_type for enums."""
    mock_node = MagicMock()
    mock_node.type = "enum_item"

    result = rust_parser._get_node_type(mock_node)
    assert result == "enum"


def test_get_node_type_trait(rust_parser):
    """Test _get_node_type for traits."""
    mock_node = MagicMock()
    mock_node.type = "trait_item"

    result = rust_parser._get_node_type(mock_node)
    assert result == "trait"


def test_get_node_type_impl(rust_parser):
    """Test _get_node_type for impl blocks."""
    mock_node = MagicMock()
    mock_node.type = "impl_item"

    result = rust_parser._get_node_type(mock_node)
    assert result == "impl"


def test_get_node_type_unknown(rust_parser):
    """Test _get_node_type for unknown types."""
    mock_node = MagicMock()
    mock_node.type = "unknown_type"

    # Mock _get_parent_scope
    original_get_parent_scope = rust_parser._get_parent_scope
    rust_parser._get_parent_scope = lambda n: None

    result = rust_parser._get_node_type(mock_node)
    assert result == "unknown_type"

    rust_parser._get_parent_scope = original_get_parent_scope


def test_parse_block_comment(rust_parser):
    """Test _parse_block_comment."""
    comment = """/**
 * This is a doc comment
 * with multiple lines
 */"""
    result = rust_parser._parse_block_comment(comment)
    assert "This is a doc comment" in result
    assert "with multiple lines" in result


def test_parse_block_comment_inner(rust_parser):
    """Test _parse_block_comment with inner doc comment."""
    comment = """/*!
 * Inner doc comment
 */"""
    result = rust_parser._parse_block_comment(comment)
    assert "Inner doc comment" in result


def test_parse_block_comment_no_stars(rust_parser):
    """Test _parse_block_comment without leading stars."""
    comment = """/**
This is clean
No leading stars
*/"""
    result = rust_parser._parse_block_comment(comment)
    assert "This is clean" in result
    assert "No leading stars" in result


def test_get_signature_non_function(rust_parser):
    """Test _get_signature returns None for non-functions."""
    mock_node = MagicMock()
    mock_node.type = "struct_item"

    result = rust_parser._get_signature(mock_node, b"")
    assert result is None


def test_get_signature_with_body(rust_parser):
    """Test _get_signature with function body."""
    code = b"fn test() { body }"
    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.start_byte = 0

    mock_body = MagicMock()
    mock_body.start_byte = 10
    mock_node.child_by_field_name = MagicMock(return_value=mock_body)

    result = rust_parser._get_signature(mock_node, code)
    assert result == "fn test()"


def test_get_signature_without_body(rust_parser):
    """Test _get_signature without function body (trait method)."""
    code = b"fn test();"
    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.start_byte = 0
    mock_node.end_byte = len(code)
    mock_node.child_by_field_name = MagicMock(return_value=None)

    result = rust_parser._get_signature(mock_node, code)
    assert result == "fn test();"


def test_get_parent_scope_impl(rust_parser):
    """Test _get_parent_scope finds impl block."""
    mock_name = MagicMock()
    mock_name.text = b"MyStruct"

    mock_impl = MagicMock()
    mock_impl.type = "impl_item"
    mock_impl.child_by_field_name = MagicMock(return_value=mock_name)

    mock_node = MagicMock()
    mock_node.parent = mock_impl

    result = rust_parser._get_parent_scope(mock_node)
    assert result == "MyStruct"


def test_get_parent_scope_trait(rust_parser):
    """Test _get_parent_scope finds trait."""
    mock_name = MagicMock()
    mock_name.text = b"MyTrait"

    mock_trait = MagicMock()
    mock_trait.type = "trait_item"
    mock_trait.child_by_field_name = MagicMock(return_value=mock_name)

    mock_node = MagicMock()
    mock_node.parent = mock_trait

    result = rust_parser._get_parent_scope(mock_node)
    assert result == "MyTrait"


def test_get_parent_scope_none(rust_parser):
    """Test _get_parent_scope returns None for top-level."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = rust_parser._get_parent_scope(mock_node)
    assert result is None


def test_get_parent_scope_declaration_list(rust_parser):
    """Test _get_parent_scope traverses through declaration_list."""
    mock_name = MagicMock()
    mock_name.text = b"MyStruct"

    mock_impl = MagicMock()
    mock_impl.type = "impl_item"
    mock_impl.child_by_field_name = MagicMock(return_value=mock_name)

    mock_decl_list = MagicMock()
    mock_decl_list.type = "declaration_list"
    mock_decl_list.parent = mock_impl

    mock_node = MagicMock()
    mock_node.parent = mock_decl_list

    result = rust_parser._get_parent_scope(mock_node)
    assert result == "MyStruct"


def test_is_async_true(rust_parser):
    """Test _is_async detects async functions."""
    mock_async = MagicMock()
    mock_async.type = "async"

    mock_modifiers = MagicMock()
    mock_modifiers.type = "function_modifiers"
    mock_modifiers.children = [mock_async]

    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.children = [mock_modifiers]

    result = rust_parser._is_async(mock_node)
    assert result is True


def test_is_async_false(rust_parser):
    """Test _is_async returns False for sync functions."""
    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.children = []

    result = rust_parser._is_async(mock_node)
    assert result is False


def test_is_unsafe_function_true(rust_parser):
    """Test _is_unsafe detects unsafe functions."""
    mock_unsafe = MagicMock()
    mock_unsafe.type = "unsafe"

    mock_modifiers = MagicMock()
    mock_modifiers.type = "function_modifiers"
    mock_modifiers.children = [mock_unsafe]

    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.children = [mock_modifiers]

    result = rust_parser._is_unsafe(mock_node)
    assert result is True


def test_is_unsafe_direct_child(rust_parser):
    """Test _is_unsafe with unsafe as direct child."""
    mock_unsafe = MagicMock()
    mock_unsafe.type = "unsafe"

    mock_node = MagicMock()
    mock_node.type = "impl_item"
    mock_node.children = [mock_unsafe]

    result = rust_parser._is_unsafe(mock_node)
    assert result is True


def test_is_unsafe_false(rust_parser):
    """Test _is_unsafe returns False for safe items."""
    mock_node = MagicMock()
    mock_node.type = "function_item"
    mock_node.children = []

    result = rust_parser._is_unsafe(mock_node)
    assert result is False


def test_is_pub_true(rust_parser):
    """Test _is_pub detects public items."""
    mock_vis = MagicMock()
    mock_vis.type = "visibility_modifier"

    mock_node = MagicMock()
    mock_node.children = [mock_vis]

    result = rust_parser._is_pub(mock_node)
    assert result is True


def test_is_pub_false(rust_parser):
    """Test _is_pub returns False for private items."""
    mock_node = MagicMock()
    mock_node.children = []

    result = rust_parser._is_pub(mock_node)
    assert result is False


def test_get_attributes(rust_parser):
    """Test _get_attributes extracts attributes."""
    mock_attr1 = MagicMock()
    mock_attr1.type = "attribute_item"
    mock_attr1.text = b"#[derive(Debug)]"

    mock_attr2 = MagicMock()
    mock_attr2.type = "attribute_item"
    mock_attr2.text = b"#[allow(unused)]"

    mock_node = MagicMock()

    mock_parent = MagicMock()
    mock_parent.children = [mock_attr1, mock_attr2, mock_node]
    mock_node.parent = mock_parent

    result = rust_parser._get_attributes(mock_node, b"")
    assert "#[derive(Debug)]" in result
    assert "#[allow(unused)]" in result


def test_get_attributes_with_doc_comments(rust_parser):
    """Test _get_attributes skips doc comments."""
    mock_comment = MagicMock()
    mock_comment.type = "line_comment"
    mock_comment.text = b"/// Doc comment"

    mock_attr = MagicMock()
    mock_attr.type = "attribute_item"
    mock_attr.text = b"#[derive(Debug)]"

    mock_node = MagicMock()

    mock_parent = MagicMock()
    mock_parent.children = [mock_attr, mock_comment, mock_node]
    mock_node.parent = mock_parent

    result = rust_parser._get_attributes(mock_node, b"")
    assert "#[derive(Debug)]" in result
    assert len(result) == 1


def test_get_attributes_no_parent(rust_parser):
    """Test _get_attributes with no parent."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = rust_parser._get_attributes(mock_node, b"")
    assert result == []


def test_get_path_name_scoped_identifier(rust_parser):
    """Test _get_path_name with scoped_identifier."""
    mock_name = MagicMock()
    mock_name.text = b"Read"

    mock_path = MagicMock()
    mock_path.type = "scoped_identifier"
    mock_path.child_by_field_name = MagicMock(return_value=mock_name)

    result = rust_parser._get_path_name(mock_path)
    assert result == "Read"


def test_get_path_name_simple(rust_parser):
    """Test _get_path_name with simple identifier."""
    mock_path = MagicMock()
    mock_path.type = "identifier"
    mock_path.text = b"MyType"
    mock_path.child_by_field_name = MagicMock(return_value=None)

    result = rust_parser._get_path_name(mock_path)
    assert result == "MyType"


def test_get_documentation_no_parent(rust_parser):
    """Test _get_documentation with no parent."""
    mock_node = MagicMock()
    mock_node.parent = None

    result = rust_parser._get_documentation(mock_node, b"")
    assert result is None


def test_get_documentation_node_not_found(rust_parser):
    """Test _get_documentation when node not found in parent."""
    mock_parent = MagicMock()
    mock_parent.children = []

    mock_node = MagicMock()
    mock_node.parent = mock_parent

    result = rust_parser._get_documentation(mock_node, b"")
    assert result is None


def test_process_match_no_def_nodes(rust_parser):
    """Test process_match with no def nodes."""
    match = {}
    result = rust_parser.process_match(match, b"")
    assert result is None


def test_process_match_no_name(rust_parser):
    """Test process_match with def but no name."""
    mock_node = MagicMock()
    match = {"def": [mock_node]}

    result = rust_parser.process_match(match, b"")
    assert result is None


def test_process_match_with_alias(rust_parser):
    """Test process_match uses alias when present."""
    code = b"use std::io::Read as MyRead;"

    mock_alias = MagicMock()
    mock_alias.text = b"MyRead"

    mock_def = MagicMock()
    mock_def.type = "use_declaration"
    mock_def.start_byte = 0
    mock_def.end_byte = len(code)
    mock_def.start_point = (0, 0)
    mock_def.end_point = (0, len(code))
    mock_def.parent = None
    mock_def.children = []

    match = {"def": [mock_def], "alias": [mock_alias]}

    result = rust_parser.process_match(match, code)
    assert result is not None
    assert result[1]["node_name"] == "MyRead"


def test_language_property(rust_parser):
    """Test that language property is set correctly."""
    assert rust_parser.language == "rust"


def test_all_results_have_required_fields(rust_parser):
    """Test that all parsed results have required fields."""
    code = """fn func() {}
struct MyStruct {}
const VALUE: i32 = 42;
"""
    results = list(rust_parser.parse(code))

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


def test_extra_always_has_required_fields(rust_parser):
    """Test that extra always contains required fields."""
    code = """fn func() {}
"""
    results = list(rust_parser.parse(code))

    extra = results[0][1]["extra"]
    assert "attributes" in extra
    assert "is_async" in extra
    assert "is_unsafe" in extra
    assert "is_pub" in extra


def test_complex_trait_impl(rust_parser):
    """Test parsing complex trait implementation."""
    code = """impl<T> Display for Container<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.value)
    }
}
"""
    results = list(rust_parser.parse(code))

    # The query might not capture impl blocks with 'for' trait,
    # but should find the method
    method_results = [r for r in results if r[1]["node_type"] == "method"]
    assert len(method_results) >= 1
    assert method_results[0][1]["node_name"] == "fmt"


def test_nested_modules(rust_parser):
    """Test parsing nested modules."""
    code = """mod outer {
    pub mod inner {
        pub fn nested_func() {}
    }
}
"""
    results = list(rust_parser.parse(code))

    mod_results = [r for r in results if r[1]["node_type"] == "module"]
    # Should find at least the outer module
    assert len(mod_results) >= 1


def test_regular_comment_breaks_doc_chain(rust_parser):
    """Test that regular comments break doc comment chains."""
    code = """/// Doc comment
// Regular comment breaks the chain
fn func() {}
"""
    results = list(rust_parser.parse(code))

    # Should not include the doc comment because regular comment broke the chain
    assert len(results) == 1
    # Documentation should be None or not include the first line
    doc = results[0][1]["documentation"]
    assert doc is None or "Doc comment" not in doc


def test_regular_block_comment_breaks_doc_chain(rust_parser):
    """Test that regular block comments break doc comment chains."""
    code = """/** Doc comment */
/* Regular block comment */
fn func() {}
"""
    results = list(rust_parser.parse(code))

    # Should not include the doc comment
    assert len(results) == 1
    doc = results[0][1]["documentation"]
    assert doc is None or "Doc comment" not in doc


def test_get_attributes_node_not_in_parent(rust_parser):
    """Test _get_attributes when node is not found in parent's children."""
    mock_other = MagicMock()

    mock_parent = MagicMock()
    mock_parent.children = [mock_other]  # Node not in list

    mock_node = MagicMock()
    mock_node.parent = mock_parent

    result = rust_parser._get_attributes(mock_node, b"")
    assert result == []
