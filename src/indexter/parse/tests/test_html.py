from pathlib import Path

import pytest

from indexter.parse.html import HtmlParser
from indexter.parse.models import Kind

FIXTURES = Path(__file__).parent / "fixtures" / "html"


@pytest.fixture
def parser():
    return HtmlParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.html").read_text()
        result = parser.parse("html/sample.html", content)
        sections = {n.name: n.signature for n in result.nodes if n.kind == Kind.SECTION}
        assert sections["Sample Page"] == "h1"
        assert sections["table"] == "table"
        assert sections["ul-list"] == "ul"

    def test_all_nodes_are_section_or_file(self, parser):
        content = (FIXTURES / "sample.html").read_text()
        result = parser.parse("html/sample.html", content)
        assert {n.kind for n in result.nodes} <= {Kind.FILE, Kind.SECTION}

    def test_no_references(self, parser):
        content = (FIXTURES / "sample.html").read_text()
        result = parser.parse("html/sample.html", content)
        assert result.refs == []

    def test_heading_nested_in_semantic_container_gets_scope(self, parser):
        result = parser.parse(
            "html/nested.html",
            "<article><h2>Inside</h2></article>",
        )
        node = next(n for n in result.nodes if n.name == "Inside")
        assert node.scope_path == ("article",)

    def test_ordered_list_named(self, parser):
        result = parser.parse("html/ol.html", "<ol><li>one</li></ol>")
        node = next(n for n in result.nodes if n.kind == Kind.SECTION)
        assert node.name == "ol-list"
        assert node.signature == "ol"

    def test_heading_with_no_text_falls_back_to_tag(self, parser):
        result = parser.parse("html/blank.html", "<h2></h2>")
        node = next(n for n in result.nodes if n.kind == Kind.SECTION)
        assert node.name == "h2"


class TestEdgeCases:
    def test_empty_file(self, parser):
        result = parser.parse("html/empty.html", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_no_semantic_elements(self, parser):
        result = parser.parse("html/plain.html", "<p>hello</p>")
        assert [n.kind for n in result.nodes] == [Kind.FILE]


class TestDefensiveBranches:
    def test_process_definition_match_without_recognized_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_scope_segment_ignores_non_element_ancestor(self, parser):
        class FakeNode:
            type = "document"

        assert parser.scope_segment(FakeNode()) is None

    def test_scope_segment_ignores_element_with_no_tag_name(self, parser):
        class FakeStartTag:
            type = "start_tag"
            children = []

        class FakeElement:
            type = "element"
            children = [FakeStartTag()]

        assert parser.scope_segment(FakeElement()) is None

    def test_scope_segment_ignores_unrecognized_tag(self, parser):
        result = parser.parse("html/div.html", "<span><h2>X</h2></span>")
        node = next(n for n in result.nodes if n.name == "X")
        assert node.scope_path == ()
