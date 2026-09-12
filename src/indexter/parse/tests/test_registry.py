"""Confirms importing indexter.parse registers every language parser, so
parse_file() dispatches correctly without callers importing each module.
"""

from indexter.parse.base import parse_file
from indexter.parse.models import Kind


def test_python_extension_registered():
    result = parse_file("a.py", "def foo():\n    pass\n")
    assert Kind.FUNCTION in {n.kind for n in result.nodes}


def test_javascript_extensions_registered():
    for path in ("a.js", "a.jsx"):
        result = parse_file(path, "function foo() {}\n")
        assert Kind.FUNCTION in {n.kind for n in result.nodes}


def test_typescript_extension_registered():
    result = parse_file("a.ts", "function foo(): void {}\n")
    assert Kind.FUNCTION in {n.kind for n in result.nodes}


def test_rust_extension_registered():
    result = parse_file("a.rs", "fn foo() {}\n")
    assert Kind.FUNCTION in {n.kind for n in result.nodes}


def test_markdown_extension_registered():
    result = parse_file("a.md", "# Title\n")
    assert Kind.SECTION in {n.kind for n in result.nodes}


def test_json_extension_registered():
    result = parse_file("a.json", '{"a": {"b": 1}}')
    assert Kind.DATA in {n.kind for n in result.nodes}


def test_yaml_extensions_registered():
    for path in ("a.yaml", "a.yml"):
        result = parse_file(path, "a:\n  b: 1\n")
        assert Kind.DATA in {n.kind for n in result.nodes}


def test_toml_extension_registered():
    result = parse_file("a.toml", "a = 1\n")
    assert Kind.DATA in {n.kind for n in result.nodes}


def test_html_extensions_registered():
    for path in ("a.html", "a.htm"):
        result = parse_file(path, "<h1>Hi</h1>\n")
        assert Kind.SECTION in {n.kind for n in result.nodes}


def test_css_extension_registered():
    result = parse_file("a.css", "body { color: red; }\n")
    assert Kind.SECTION in {n.kind for n in result.nodes}


def test_unregistered_extension_falls_back_to_chunk():
    result = parse_file("a.unknown-ext", "plain text\n")
    assert {n.kind for n in result.nodes} == {Kind.FILE, Kind.CHUNK}
