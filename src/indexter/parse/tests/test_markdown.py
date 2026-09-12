from pathlib import Path

import pytest

from indexter.parse.markdown import MarkdownParser
from indexter.parse.models import Kind

FIXTURES = Path(__file__).parent / "fixtures" / "markdown"


@pytest.fixture
def parser():
    return MarkdownParser()


class TestNodes:
    def test_sample_fixture_headings(self, parser):
        content = (FIXTURES / "sample.md").read_text()
        result = parser.parse("markdown/sample.md", content)
        names = {n.name for n in result.nodes if n.kind == Kind.SECTION}
        assert names == {
            "Indexter Fixture",
            "Indexter Fixture > Setup",
            "Indexter Fixture > Setup > Prerequisites",
            "Indexter Fixture > Usage",
            "Indexter Fixture > Usage > Basic usage",
            "Indexter Fixture > Usage > Advanced usage",
        }

    def test_all_sections_have_a_level_signature(self, parser):
        content = (FIXTURES / "sample.md").read_text()
        result = parser.parse("markdown/sample.md", content)
        sections = [n for n in result.nodes if n.kind == Kind.SECTION]
        assert sections
        assert all(n.signature in ("h1", "h2", "h3") for n in sections)

    def test_nested_heading_scope_path_is_empty_path_is_in_name(self, parser):
        content = (FIXTURES / "sample.md").read_text()
        result = parser.parse("markdown/sample.md", content)
        nested = next(n for n in result.nodes if n.name.endswith("Prerequisites"))
        assert nested.scope_path == ()

    def test_no_references(self, parser):
        content = (FIXTURES / "sample.md").read_text()
        result = parser.parse("markdown/sample.md", content)
        assert result.refs == []

    def test_sections_nest_by_byte_range(self, parser):
        content = (FIXTURES / "sample.md").read_text()
        result = parser.parse("markdown/sample.md", content)
        setup = next(n for n in result.nodes if n.name == "Indexter Fixture > Setup")
        prereq = next(n for n in result.nodes if n.name.endswith("Prerequisites"))
        assert setup.start_byte <= prereq.start_byte
        assert prereq.end_byte <= setup.end_byte
        assert prereq.parent_id == setup.id


class TestEdgeCases:
    def test_empty_file(self, parser):
        result = parser.parse("markdown/empty.md", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_no_headings(self, parser):
        result = parser.parse("markdown/plain.md", "just a paragraph\n")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_setext_heading_is_not_captured(self, parser):
        # Setext (underline-style) headings aren't matched by the ATX-only
        # query -- only the file node should appear.
        result = parser.parse("markdown/setext.md", "Title\n=====\n")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_heading_with_no_text_is_skipped(self, parser):
        # `#` with nothing after it has no `inline` child at all.
        result = parser.parse("markdown/blank_heading.md", "#\n")
        assert [n.kind for n in result.nodes] == [Kind.FILE]


class TestDefensiveBranches:
    """Real markdown that reaches an ERROR-containing atx_heading is hard
    to construct -- exercised directly via a minimal node double."""

    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_error_heading_is_skipped(self, parser):
        class FakeHeading:
            type = "atx_heading"
            has_error = True
            children = []

        assert parser.process_definition_match({"def": [FakeHeading()]}, b"") is None

    def test_heading_with_no_wrapping_section_falls_back_to_its_own_range(self, parser):
        # Every real tree-sitter-markdown heading is wrapped in a `section`
        # node -- exercised directly since real source can't reach it.
        class FakeMarker:
            type = "atx_h1_marker"

        class FakeInline:
            type = "inline"
            text = b"Title"

        class FakeHeading:
            type = "atx_heading"
            has_error = False
            children = [FakeMarker(), FakeInline()]
            parent = None
            start_point = (0, 0)
            end_point = (0, 7)
            start_byte = 0
            end_byte = 7

        node = parser.process_definition_match({"def": [FakeHeading()]}, b"")
        assert node is not None
        assert node.end_byte == 7
