from pathlib import Path

import pytest

from indexter.parse.models import Kind, RefKind
from indexter.parse.python import PythonParser

FIXTURES = Path(__file__).parent / "fixtures" / "python"


@pytest.fixture
def parser():
    return PythonParser()


def _by_name(nodes, name):
    return next(n for n in nodes if n.name == name)


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        kinds = {n.name: n.kind for n in result.nodes if n.name}
        assert kinds["MAX_RETRIES"] == Kind.CONSTANT
        assert kinds["Base"] == Kind.CLASS
        assert kinds["Handler"] == Kind.CLASS
        assert kinds["login"] == Kind.METHOD
        assert kinds["validate"] == Kind.METHOD
        assert kinds["deprecated"] == Kind.FUNCTION
        assert kinds["standalone"] == Kind.FUNCTION

    def test_no_import_nodes(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        assert all(n.kind != "import" for n in result.nodes)
        # Only the expected symbols show up -- nothing for the import statements.
        names = {n.name for n in result.nodes}
        assert "os" not in names
        assert "sibling" not in names

    def test_decorated_function_yields_one_node_including_decorator(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        matches = [n for n in result.nodes if n.name == "standalone"]
        assert len(matches) == 1
        node = matches[0]
        source_bytes = content.encode()
        text = source_bytes[node.start_byte : node.end_byte].decode()
        assert text.startswith("@deprecated")
        assert "def standalone" in text

    def test_byte_ranges_slice_back_to_exact_source(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        source_bytes = content.encode()
        login = _by_name(result.nodes, "login")
        text = source_bytes[login.start_byte : login.end_byte].decode()
        assert text.startswith("def login(self, user):")
        assert "self.validate(user)" in text

    def test_documented_class_with_base(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        handler = _by_name(result.nodes, "Handler")
        assert handler.docstring == "Handles requests, delegating validation to a helper."
        [base_ref] = [r for r in result.refs if r.ref_kind == RefKind.INHERITS]
        assert base_ref.raw_name == "Base"
        assert base_ref.from_node_id == handler.id

    def test_module_constant(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        constant = _by_name(result.nodes, "MAX_RETRIES")
        assert constant.kind == Kind.CONSTANT
        assert constant.parent_id == _by_name(result.nodes, "").id

    def test_non_constant_assignment_is_not_a_node(self, parser):
        result = parser.parse("a.py", "lowercase_var = 5\n")
        assert all(n.name != "lowercase_var" for n in result.nodes)

    def test_empty_file(self, parser):
        result = parser.parse("a.py", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]
        assert result.errors == []


class TestCollisions:
    def test_nested_function_distinct_from_file_scope(self, parser):
        content = """
def inner():
    return "top"


def outer():
    def inner():
        return "nested"

    return inner
"""
        result = parser.parse("a.py", content)
        top_level = [n for n in result.nodes if n.name == "inner" and n.scope_path == ()]
        nested = [n for n in result.nodes if n.name == "inner" and n.scope_path == ("outer",)]
        assert len(top_level) == 1
        assert len(nested) == 1
        assert top_level[0].id != nested[0].id

    def test_two_inners_under_different_parents_are_distinct(self, parser):
        content = (FIXTURES / "scopes.py").read_text()
        result = parser.parse("python/scopes.py", content)
        inners = [n for n in result.nodes if n.name == "inner"]
        assert len(inners) == 2
        assert inners[0].scope_path == ("outer_a",)
        assert inners[1].scope_path == ("outer_b",)
        assert inners[0].id != inners[1].id
        assert "~" not in inners[0].id
        assert "~" not in inners[1].id

    def test_duplicate_login_gets_suffixed(self, parser):
        content = (FIXTURES / "scopes.py").read_text()
        result = parser.parse("python/scopes.py", content)
        logins = sorted((n for n in result.nodes if n.name == "login"), key=lambda n: n.start_line)
        assert len(logins) == 2
        assert logins[0].id == "python/scopes.py::login#function"
        assert logins[1].id == "python/scopes.py::login#function~2"


class TestReferences:
    def test_self_attribute_call_head(self, parser):
        content = "class C:\n    def m(self):\n        self.validate(1)\n"
        result = parser.parse("a.py", content)
        [ref] = [r for r in result.refs if r.ref_kind == RefKind.CALLS]
        assert ref.raw_name == "self.validate"
        assert ref.head == "self"

    def test_nested_attribute_call_head(self, parser):
        content = "def f():\n    os.path.join('a')\n"
        result = parser.parse("a.py", content)
        [ref] = result.refs
        assert ref.raw_name == "os.path.join"
        assert ref.head == "os"

    def test_bare_call_head_equals_raw_name(self, parser):
        result = parser.parse("a.py", "def f():\n    helper()\n")
        [ref] = result.refs
        assert ref.raw_name == "helper"
        assert ref.head == "helper"

    def test_relative_import_dots_preserved(self, parser):
        result = parser.parse("a.py", "from . import sibling\nfrom ..pkg import thing\n")
        raw_names = {r.raw_name for r in result.refs if r.ref_kind == RefKind.IMPORTS}
        assert raw_names == {".sibling", "..pkg.thing"}

    def test_plain_and_from_imports(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        raw_names = {r.raw_name for r in result.refs if r.ref_kind == RefKind.IMPORTS}
        assert "os" in raw_names
        assert "collections.OrderedDict" in raw_names

    def test_multiple_bases_yield_one_ref_each(self, parser):
        result = parser.parse("a.py", "class C(A, B):\n    pass\n")
        inherits = [r for r in result.refs if r.ref_kind == RefKind.INHERITS]
        assert {r.raw_name for r in inherits} == {"A", "B"}
        assert len(inherits) == 2

    def test_every_ref_origin_matches_a_node_from_the_same_parse(self, parser):
        content = (FIXTURES / "sample.py").read_text()
        result = parser.parse("python/sample.py", content)
        node_ids = {n.id for n in result.nodes}
        assert result.refs  # sanity: there are references to check
        assert all(r.from_node_id in node_ids for r in result.refs)


class TestSyntaxTolerance:
    def test_syntax_error_still_yields_good_constructs(self, parser):
        result = parser.parse("a.py", "def good():\n    return 1\n\ndef bad(:::\n")
        names = {n.name for n in result.nodes}
        assert "good" in names


class TestDocstringEdgeCases:
    def test_non_string_first_statement_has_no_docstring(self, parser):
        result = parser.parse("a.py", "def f():\n    1\n    return 1\n")
        node = next(n for n in result.nodes if n.name == "f")
        assert node.docstring is None

    def test_docstring_with_no_body_or_children(self, parser):
        class FakeNode:
            type = "function_definition"

            def child_by_field_name(self, name):
                return None

        assert parser._docstring(FakeNode(), b"") is None

    def test_docstring_wrapped_in_expression_statement(self, parser):
        # This tree-sitter-python grammar version emits bare strings as
        # direct block children (the common case, handled above), not
        # wrapped in expression_statement -- exercised directly since real
        # source can't reach it with the grammar in use.
        class FakeString:
            type = "string"
            text = b'"""wrapped"""'

        class FakeExprStatement:
            type = "expression_statement"
            children = [FakeString()]

        class FakeBody:
            children = [FakeExprStatement()]

        class FakeNode:
            type = "function_definition"

            def child_by_field_name(self, name):
                return FakeBody() if name == "body" else None

        assert parser._docstring(FakeNode(), b"") == "wrapped"


class TestDefensiveBranches:
    """Match-handler fallbacks real tree-sitter queries never actually
    produce -- exercised directly since every capture group our queries
    emit is already handled."""

    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_process_definition_match_without_name_capture(self, parser):
        class FakeNode:
            type = "function_definition"
            parent = None

        assert parser.process_definition_match({"def": [FakeNode()]}, b"") is None

    def test_process_reference_match_with_no_recognized_capture(self, parser):
        assert parser.process_reference_match({}, b"") is None
