"""Tests for parsers __init__.py module."""

from pathlib import Path

import pytest

from indexter.parsers import (
    EXT_TO_LANGUAGE_PARSER,
    get_parser,
)
from indexter.parsers.base import BaseParser
from indexter.parsers.chunk import ChunkParser
from indexter.parsers.css import CssParser
from indexter.parsers.html import HtmlParser
from indexter.parsers.javascript import JavaScriptParser
from indexter.parsers.json import JsonParser
from indexter.parsers.markdown import MarkdownParser
from indexter.parsers.python import PythonParser
from indexter.parsers.rust import RustParser
from indexter.parsers.toml import TomlParser
from indexter.parsers.typescript import TypeScriptParser
from indexter.parsers.yaml import YamlParser

# Tests for EXT_TO_LANGUAGE_PARSER mapping


def test_ext_to_language_parser_exists():
    """Test that EXT_TO_LANGUAGE_PARSER dict exists."""
    assert EXT_TO_LANGUAGE_PARSER is not None
    assert isinstance(EXT_TO_LANGUAGE_PARSER, dict)


def test_ext_to_language_parser_css():
    """Test CSS extension mapping."""
    assert EXT_TO_LANGUAGE_PARSER[".css"] == CssParser


def test_ext_to_language_parser_html():
    """Test HTML extension mapping."""
    assert EXT_TO_LANGUAGE_PARSER[".html"] == HtmlParser


def test_ext_to_language_parser_javascript():
    """Test JavaScript extension mappings."""
    assert EXT_TO_LANGUAGE_PARSER[".js"] == JavaScriptParser
    assert EXT_TO_LANGUAGE_PARSER[".jsx"] == JavaScriptParser


def test_ext_to_language_parser_json():
    """Test JSON extension mapping."""
    assert EXT_TO_LANGUAGE_PARSER[".json"] == JsonParser


def test_ext_to_language_parser_markdown():
    """Test Markdown extension mappings."""
    assert EXT_TO_LANGUAGE_PARSER[".md"] == MarkdownParser
    assert EXT_TO_LANGUAGE_PARSER[".mkd"] == MarkdownParser
    assert EXT_TO_LANGUAGE_PARSER[".markdown"] == MarkdownParser


def test_ext_to_language_parser_python():
    """Test Python extension mapping."""
    assert EXT_TO_LANGUAGE_PARSER[".py"] == PythonParser


def test_ext_to_language_parser_rust():
    """Test Rust extension mapping."""
    assert EXT_TO_LANGUAGE_PARSER[".rs"] == RustParser


def test_ext_to_language_parser_toml():
    """Test TOML extension mapping."""
    assert EXT_TO_LANGUAGE_PARSER[".toml"] == TomlParser


def test_ext_to_language_parser_typescript():
    """Test TypeScript extension mappings."""
    assert EXT_TO_LANGUAGE_PARSER[".ts"] == TypeScriptParser
    assert EXT_TO_LANGUAGE_PARSER[".tsx"] == TypeScriptParser


def test_ext_to_language_parser_yaml():
    """Test YAML extension mappings."""
    assert EXT_TO_LANGUAGE_PARSER[".yaml"] == YamlParser
    assert EXT_TO_LANGUAGE_PARSER[".yml"] == YamlParser


def test_ext_to_language_parser_all_values_are_classes():
    """Test that all values in mapping are parser classes."""
    for _, parser_cls in EXT_TO_LANGUAGE_PARSER.items():
        assert isinstance(parser_cls, type)
        assert issubclass(parser_cls, BaseParser)


def test_ext_to_language_parser_all_extensions_lowercase():
    """Test that all extension keys are lowercase."""
    for ext in EXT_TO_LANGUAGE_PARSER.keys():
        assert ext == ext.lower()
        assert ext.startswith(".")


# Tests for get_parser function


def test_get_parser_css():
    """Test get_parser returns CssParser for .css files."""
    parser = get_parser("styles.css")
    assert isinstance(parser, CssParser)


def test_get_parser_html():
    """Test get_parser returns HtmlParser for .html files."""
    parser = get_parser("index.html")
    assert isinstance(parser, HtmlParser)


def test_get_parser_javascript():
    """Test get_parser returns JavaScriptParser for .js files."""
    parser = get_parser("app.js")
    assert isinstance(parser, JavaScriptParser)


def test_get_parser_jsx():
    """Test get_parser returns JavaScriptParser for .jsx files."""
    parser = get_parser("component.jsx")
    assert isinstance(parser, JavaScriptParser)


def test_get_parser_json():
    """Test get_parser returns JsonParser for .json files."""
    parser = get_parser("config.json")
    assert isinstance(parser, JsonParser)


def test_get_parser_markdown():
    """Test get_parser returns MarkdownParser for .md files."""
    parser = get_parser("README.md")
    assert isinstance(parser, MarkdownParser)


def test_get_parser_markdown_mkd():
    """Test get_parser returns MarkdownParser for .mkd files."""
    parser = get_parser("doc.mkd")
    assert isinstance(parser, MarkdownParser)


def test_get_parser_markdown_full():
    """Test get_parser returns MarkdownParser for .markdown files."""
    parser = get_parser("document.markdown")
    assert isinstance(parser, MarkdownParser)


def test_get_parser_python():
    """Test get_parser returns PythonParser for .py files."""
    parser = get_parser("script.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_rust():
    """Test get_parser returns RustParser for .rs files."""
    parser = get_parser("main.rs")
    assert isinstance(parser, RustParser)


def test_get_parser_toml():
    """Test get_parser returns TomlParser for .toml files."""
    parser = get_parser("config.toml")
    assert isinstance(parser, TomlParser)


def test_get_parser_typescript():
    """Test get_parser returns TypeScriptParser for .ts files."""
    parser = get_parser("app.ts")
    assert isinstance(parser, TypeScriptParser)


def test_get_parser_tsx():
    """Test get_parser returns TypeScriptParser for .tsx files."""
    parser = get_parser("component.tsx")
    assert isinstance(parser, TypeScriptParser)


def test_get_parser_yaml():
    """Test get_parser returns YamlParser for .yaml files."""
    parser = get_parser("config.yaml")
    assert isinstance(parser, YamlParser)


def test_get_parser_yml():
    """Test get_parser returns YamlParser for .yml files."""
    parser = get_parser("docker-compose.yml")
    assert isinstance(parser, YamlParser)


def test_get_parser_case_insensitive():
    """Test get_parser is case-insensitive for extensions."""
    parser_lower = get_parser("file.py")
    parser_upper = get_parser("file.PY")
    parser_mixed = get_parser("file.Py")

    assert isinstance(parser_lower, PythonParser)
    assert isinstance(parser_upper, PythonParser)
    assert isinstance(parser_mixed, PythonParser)


def test_get_parser_with_path():
    """Test get_parser works with file paths."""
    parser = get_parser("/path/to/script.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_with_complex_path():
    """Test get_parser works with complex paths."""
    parser = get_parser("/home/user/projects/myapp/src/utils/helper.js")
    assert isinstance(parser, JavaScriptParser)


def test_get_parser_unknown_extension():
    """Test get_parser returns ChunkParser for unknown extensions."""
    parser = get_parser("file.txt")
    assert isinstance(parser, ChunkParser)


def test_get_parser_no_extension():
    """Test get_parser returns ChunkParser for files with no extension."""
    parser = get_parser("Makefile")
    assert isinstance(parser, ChunkParser)


def test_get_parser_multiple_dots():
    """Test get_parser handles multiple dots in filename."""
    parser = get_parser("test.config.json")
    assert isinstance(parser, JsonParser)


def test_get_parser_hidden_file():
    """Test get_parser works with hidden files."""
    parser = get_parser(".gitignore")
    assert isinstance(parser, ChunkParser)


def test_get_parser_hidden_file_with_known_extension():
    """Test get_parser works with hidden files that have known extensions."""
    parser = get_parser(".eslintrc.json")
    assert isinstance(parser, JsonParser)


def test_get_parser_relative_path():
    """Test get_parser works with relative paths."""
    parser = get_parser("./src/main.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_returns_new_instance():
    """Test get_parser returns a new instance each time."""
    parser1 = get_parser("file.py")
    parser2 = get_parser("file.py")

    assert isinstance(parser1, PythonParser)
    assert isinstance(parser2, PythonParser)
    assert parser1 is not parser2


def test_get_parser_empty_string():
    """Test get_parser with empty string returns ChunkParser."""
    parser = get_parser("")
    assert isinstance(parser, ChunkParser)


def test_get_parser_dot_only():
    """Test get_parser with just a dot returns ChunkParser."""
    parser = get_parser(".")
    assert isinstance(parser, ChunkParser)


def test_get_parser_double_extension():
    """Test get_parser uses only the last extension."""
    parser = get_parser("file.py.txt")
    assert isinstance(parser, ChunkParser)


def test_get_parser_all_supported_extensions():
    """Test get_parser for all supported extensions in mapping."""
    test_cases = {
        ".css": CssParser,
        ".html": HtmlParser,
        ".js": JavaScriptParser,
        ".jsx": JavaScriptParser,
        ".json": JsonParser,
        ".md": MarkdownParser,
        ".mkd": MarkdownParser,
        ".markdown": MarkdownParser,
        ".py": PythonParser,
        ".rs": RustParser,
        ".toml": TomlParser,
        ".ts": TypeScriptParser,
        ".tsx": TypeScriptParser,
        ".yaml": YamlParser,
        ".yml": YamlParser,
    }

    for ext, expected_class in test_cases.items():
        parser = get_parser(f"test{ext}")
        assert isinstance(parser, expected_class), f"Failed for extension {ext}"


def test_get_parser_with_pathlib_path():
    """Test get_parser works when given a pathlib Path as string."""
    path = str(Path("src") / "main.py")
    parser = get_parser(path)
    assert isinstance(parser, PythonParser)


def test_get_parser_windows_path():
    """Test get_parser works with Windows-style paths."""
    parser = get_parser(r"C:\Users\user\file.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_url_like_string():
    """Test get_parser with URL-like strings."""
    parser = get_parser("file:///home/user/script.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_uppercase_extension():
    """Test get_parser with fully uppercase extension."""
    parser = get_parser("FILE.JSON")
    assert isinstance(parser, JsonParser)


def test_get_parser_mixed_case_extension():
    """Test get_parser with mixed case extension."""
    parser = get_parser("test.JaVaScRiPt.Js")
    assert isinstance(parser, JavaScriptParser)


def test_get_parser_suffix_extraction():
    """Test that get_parser correctly extracts suffix using Path."""
    # This tests the internal use of Path(document_path).suffix.lower()
    test_files = [
        ("simple.py", PythonParser),
        ("/abs/path/to/file.rs", RustParser),
        ("./relative/path.ts", TypeScriptParser),
        ("file.backup.json", JsonParser),
        ("no_extension", ChunkParser),
    ]

    for filepath, expected_class in test_files:
        parser = get_parser(filepath)
        assert isinstance(parser, expected_class)


def test_get_parser_fallback_to_chunk_parser():
    """Test that unknown extensions fall back to ChunkParser."""
    unknown_extensions = [
        "file.txt",
        "file.doc",
        "file.pdf",
        "file.xml",
        "file.unknown",
        "file.xyz123",
    ]

    for filename in unknown_extensions:
        parser = get_parser(filename)
        assert isinstance(parser, ChunkParser), f"Failed for {filename}"


def test_get_parser_special_characters_in_path():
    """Test get_parser with special characters in path."""
    parser = get_parser("/path/with spaces/file name.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_unicode_in_path():
    """Test get_parser with unicode characters in path."""
    parser = get_parser("/path/文件/файл.py")
    assert isinstance(parser, PythonParser)


def test_get_parser_extension_only():
    """Test get_parser with just an extension as filename returns ChunkParser."""
    # .py as a filename has no extension (it's the whole name)
    parser = get_parser(".py")
    assert isinstance(parser, ChunkParser)


def test_get_parser_returns_base_parser():
    """Test that all parsers returned are BaseParser instances."""
    test_files = [
        "file.py",
        "file.js",
        "file.unknown",
        "file.yaml",
    ]

    for filename in test_files:
        parser = get_parser(filename)
        assert isinstance(parser, BaseParser)


# =============================================================================
# Integration Tests with Git Repository Fixture
# =============================================================================


def test_git_repo_fixture_creates_all_files(sample_git_repo, sample_files_mapping):
    """Test that the git repo fixture creates all expected files."""
    for filepath in sample_files_mapping.keys():
        file_path = sample_git_repo / filepath
        assert file_path.exists(), f"Expected file {filepath} was not created"


def test_git_repo_is_valid_git_repository(sample_git_repo):
    """Test that the fixture creates a valid git repository."""
    git_dir = sample_git_repo / ".git"
    assert git_dir.exists()
    assert git_dir.is_dir()


def test_get_parser_with_all_repo_files(sample_git_repo, sample_files_mapping):
    """Test get_parser returns correct parser type for all files in the repo."""
    for filepath, expected_parser_class in sample_files_mapping.items():
        full_path = str(sample_git_repo / filepath)
        parser = get_parser(full_path)
        assert isinstance(parser, expected_parser_class), (
            f"Expected {expected_parser_class.__name__} for {filepath}, got {type(parser).__name__}"
        )


def test_parsers_can_parse_repo_files(sample_git_repo, sample_files_mapping):
    """Test that parsers can successfully parse all files in the repo."""
    for filepath, _ in sample_files_mapping.items():
        full_path = sample_git_repo / filepath
        content = full_path.read_text()

        parser = get_parser(str(full_path))

        # Parse should not raise an exception
        try:
            results = list(parser.parse(content))
            # Results should be a list (possibly empty for empty files)
            assert isinstance(results, list), f"Parse failed for {filepath}"
        except Exception as e:
            pytest.fail(f"Parser raised exception for {filepath}: {e}")


def test_python_parser_extracts_definitions(sample_git_repo):
    """Test that PythonParser extracts function and class definitions."""
    file_path = sample_git_repo / "src" / "main.py"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find at least the function and class
    assert len(results) > 0

    # Check we found expected nodes
    node_names = [r[1]["node_name"] for r in results]
    assert "greet" in node_names, f"Expected 'greet' function, found: {node_names}"
    assert "Calculator" in node_names, f"Expected 'Calculator' class, found: {node_names}"


def test_javascript_parser_extracts_definitions(sample_git_repo):
    """Test that JavaScriptParser extracts function and class definitions."""
    file_path = sample_git_repo / "src" / "index.js"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find definitions
    assert len(results) > 0

    node_names = [r[1]["node_name"] for r in results]
    assert "greet" in node_names, f"Expected 'greet' function, found: {node_names}"
    assert "Calculator" in node_names, f"Expected 'Calculator' class, found: {node_names}"


def test_typescript_parser_extracts_definitions(sample_git_repo):
    """Test that TypeScriptParser extracts interface, function and class definitions."""
    file_path = sample_git_repo / "src" / "app.ts"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find definitions
    assert len(results) > 0

    node_names = [r[1]["node_name"] for r in results]
    assert "Person" in node_names, f"Expected 'Person' interface, found: {node_names}"
    assert "greet" in node_names, f"Expected 'greet' function, found: {node_names}"
    assert "Calculator" in node_names, f"Expected 'Calculator' class, found: {node_names}"


def test_rust_parser_extracts_definitions(sample_git_repo):
    """Test that RustParser extracts function and struct definitions."""
    file_path = sample_git_repo / "src" / "lib.rs"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find definitions
    assert len(results) > 0

    node_names = [r[1]["node_name"] for r in results]
    assert "greet" in node_names, f"Expected 'greet' function, found: {node_names}"
    assert "Calculator" in node_names, f"Expected 'Calculator' struct, found: {node_names}"


def test_yaml_parser_extracts_structure(sample_git_repo):
    """Test that YamlParser extracts YAML mappings and sequences."""
    file_path = sample_git_repo / "config" / "config.yaml"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find mappings/sequences
    assert len(results) > 0

    # Check node types
    node_types = [r[1]["node_type"] for r in results]
    assert "mapping" in node_types or "sequence" in node_types


def test_json_parser_extracts_structure(sample_git_repo):
    """Test that JsonParser extracts JSON structure."""
    file_path = sample_git_repo / "package.json"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find objects/arrays
    assert len(results) > 0


def test_html_parser_extracts_elements(sample_git_repo):
    """Test that HtmlParser extracts HTML elements."""
    file_path = sample_git_repo / "index.html"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find elements
    assert len(results) > 0


def test_css_parser_extracts_rules(sample_git_repo):
    """Test that CssParser extracts CSS rules."""
    file_path = sample_git_repo / "styles" / "main.css"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find rules
    assert len(results) > 0


def test_markdown_parser_extracts_structure(sample_git_repo):
    """Test that MarkdownParser extracts Markdown structure."""
    file_path = sample_git_repo / "README.md"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find headings/sections
    assert len(results) > 0


def test_toml_parser_extracts_structure(sample_git_repo):
    """Test that TomlParser extracts TOML tables and pairs."""
    file_path = sample_git_repo / "pyproject.toml"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    # Should find tables/pairs
    assert len(results) > 0

    # Check node types
    node_types = [r[1]["node_type"] for r in results]
    assert "table" in node_types or "pair" in node_types or "table_array_element" in node_types


def test_chunk_parser_handles_txt_files(sample_git_repo):
    """Test that ChunkParser handles .txt files."""
    file_path = sample_git_repo / "notes.txt"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    assert isinstance(parser, ChunkParser)

    results = list(parser.parse(content))
    # ChunkParser should return some results for non-empty content
    assert isinstance(results, list)


def test_chunk_parser_handles_unknown_extensions(sample_git_repo):
    """Test that ChunkParser handles files with unknown extensions."""
    unknown_files = ["data.csv", "config.xml", "script.sh", "Makefile", "Dockerfile"]

    for filename in unknown_files:
        file_path = sample_git_repo / filename
        content = file_path.read_text()

        parser = get_parser(str(file_path))
        assert isinstance(parser, ChunkParser), f"Expected ChunkParser for {filename}"

        # Should not raise
        results = list(parser.parse(content))
        assert isinstance(results, list)


def test_dotfiles_handled_correctly(sample_git_repo):
    """Test that dotfiles (hidden files) are handled correctly."""
    # .gitignore has no recognized extension -> ChunkParser
    parser = get_parser(str(sample_git_repo / ".gitignore"))
    assert isinstance(parser, ChunkParser)

    # .eslintrc.json has .json extension -> JsonParser
    parser = get_parser(str(sample_git_repo / ".eslintrc.json"))
    assert isinstance(parser, JsonParser)

    # .env.example has no recognized extension -> ChunkParser
    parser = get_parser(str(sample_git_repo / ".env.example"))
    assert isinstance(parser, ChunkParser)


def test_files_with_multiple_dots(sample_git_repo):
    """Test that files with multiple dots use the last extension."""
    # config.prod.json -> JsonParser
    parser = get_parser(str(sample_git_repo / "config.prod.json"))
    assert isinstance(parser, JsonParser)

    # styles.module.css -> CssParser
    parser = get_parser(str(sample_git_repo / "styles.module.css"))
    assert isinstance(parser, CssParser)

    # test.spec.ts -> TypeScriptParser
    parser = get_parser(str(sample_git_repo / "test.spec.ts"))
    assert isinstance(parser, TypeScriptParser)


def test_empty_files_dont_crash_parsers(sample_git_repo):
    """Test that empty files don't crash parsers."""
    empty_files = ["empty.py", "empty.json", "empty.yaml"]

    for filename in empty_files:
        file_path = sample_git_repo / filename
        content = file_path.read_text()

        parser = get_parser(str(file_path))

        # Should not raise
        try:
            results = list(parser.parse(content))
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Parser crashed on empty file {filename}: {e}")


def test_all_markdown_extensions_work(sample_git_repo):
    """Test that all markdown extensions (.md, .mkd, .markdown) work."""
    md_files = [
        ("README.md", MarkdownParser),
        ("docs/guide.mkd", MarkdownParser),
        ("docs/full.markdown", MarkdownParser),
    ]

    for filepath, expected_class in md_files:
        full_path = sample_git_repo / filepath
        parser = get_parser(str(full_path))
        assert isinstance(parser, expected_class), f"Wrong parser for {filepath}"

        content = full_path.read_text()
        results = list(parser.parse(content))
        assert isinstance(results, list)


def test_all_yaml_extensions_work(sample_git_repo):
    """Test that all YAML extensions (.yaml, .yml) work."""
    yaml_files = [
        "config/config.yaml",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
    ]

    for filepath in yaml_files:
        full_path = sample_git_repo / filepath
        parser = get_parser(str(full_path))
        assert isinstance(parser, YamlParser), f"Wrong parser for {filepath}"

        content = full_path.read_text()
        results = list(parser.parse(content))
        assert isinstance(results, list)


def test_all_toml_files_work(sample_git_repo):
    """Test that all TOML files work with TomlParser."""
    toml_files = [
        "pyproject.toml",
        "config/app.toml",
        "Cargo.toml",
    ]

    for filepath in toml_files:
        full_path = sample_git_repo / filepath
        parser = get_parser(str(full_path))
        assert isinstance(parser, TomlParser), f"Wrong parser for {filepath}"

        content = full_path.read_text()
        results = list(parser.parse(content))
        assert isinstance(results, list)
        assert len(results) > 0, f"Expected results for {filepath}"


def test_jsx_and_tsx_files_work(sample_git_repo):
    """Test that JSX and TSX files work with correct parsers."""
    # JSX should use JavaScriptParser
    jsx_path = sample_git_repo / "src" / "components" / "Greeting.jsx"
    parser = get_parser(str(jsx_path))
    assert isinstance(parser, JavaScriptParser)

    content = jsx_path.read_text()
    results = list(parser.parse(content))
    assert len(results) > 0

    # TSX should use TypeScriptParser
    tsx_path = sample_git_repo / "src" / "components" / "App.tsx"
    parser = get_parser(str(tsx_path))
    assert isinstance(parser, TypeScriptParser)

    content = tsx_path.read_text()
    results = list(parser.parse(content))
    assert len(results) > 0


def test_nested_directory_paths_work(sample_git_repo):
    """Test that deeply nested paths work correctly."""
    nested_files = [
        "src/utils/helpers.py",
        "src/components/Greeting.jsx",
        ".github/workflows/ci.yml",
    ]

    for filepath in nested_files:
        full_path = sample_git_repo / filepath
        parser = get_parser(str(full_path))
        assert isinstance(parser, BaseParser)

        content = full_path.read_text()
        results = list(parser.parse(content))
        assert isinstance(results, list)


def test_parser_results_have_required_fields(sample_git_repo):
    """Test that parser results contain all required NodeInfo fields."""
    required_fields = [
        "language",
        "node_type",
        "node_name",
        "start_byte",
        "end_byte",
        "start_line",
        "end_line",
        "extra",
    ]

    # Test with a Python file that should have results
    file_path = sample_git_repo / "src" / "main.py"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    assert len(results) > 0, "Expected at least one result"

    for _, info in results:
        for field in required_fields:
            assert field in info, f"Missing required field: {field}"


def test_parser_line_numbers_are_valid(sample_git_repo):
    """Test that parser line numbers are valid (1-based, positive)."""
    file_path = sample_git_repo / "src" / "main.py"
    content = file_path.read_text()

    parser = get_parser(str(file_path))
    results = list(parser.parse(content))

    for _, info in results:
        assert info["start_line"] >= 1, "start_line should be >= 1"
        assert info["end_line"] >= info["start_line"], "end_line should be >= start_line"
        assert info["start_byte"] >= 0, "start_byte should be >= 0"
        assert info["end_byte"] >= info["start_byte"], "end_byte should be >= start_byte"
