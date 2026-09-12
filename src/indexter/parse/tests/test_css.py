from pathlib import Path

import pytest

from indexter.parse.css import CssParser
from indexter.parse.models import Kind

FIXTURES = Path(__file__).parent / "fixtures" / "css"


@pytest.fixture
def parser():
    return CssParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.css").read_text()
        result = parser.parse("css/sample.css", content)
        by_name_scope = {(n.name, n.scope_path) for n in result.nodes if n.kind == Kind.SECTION}
        assert ("body", ()) in by_name_scope
        assert ("@media", ()) in by_name_scope
        assert ("body", ("@media",)) in by_name_scope

    def test_all_nodes_are_section_or_file(self, parser):
        content = (FIXTURES / "sample.css").read_text()
        result = parser.parse("css/sample.css", content)
        assert {n.kind for n in result.nodes} <= {Kind.FILE, Kind.SECTION}

    def test_no_references(self, parser):
        content = (FIXTURES / "sample.css").read_text()
        result = parser.parse("css/sample.css", content)
        assert result.refs == []

    def test_nested_rule_parents_to_media_rule(self, parser):
        content = (FIXTURES / "sample.css").read_text()
        result = parser.parse("css/sample.css", content)
        media = next(n for n in result.nodes if n.name == "@media")
        nested_body = next(n for n in result.nodes if n.name == "body" and n.scope_path == ("@media",))
        assert nested_body.parent_id == media.id

    def test_keyframes_named_with_identifier(self, parser):
        result = parser.parse("css/kf.css", "@keyframes spin { from { opacity: 0; } to { opacity: 1; } }")
        node = next(n for n in result.nodes if n.kind == Kind.SECTION)
        assert node.name == "@keyframes spin"

    def test_native_nested_rule_gets_selector_scope(self, parser):
        result = parser.parse("css/nest.css", ".a { .b { color: red; } }\n")
        outer = next(n for n in result.nodes if n.name == ".a")
        inner = next(n for n in result.nodes if n.name == ".b")
        assert inner.scope_path == (".a",)
        assert inner.parent_id == outer.id

    def test_rule_nested_in_unrecognized_at_rule_gets_its_keyword_as_scope(self, parser):
        result = parser.parse("css/unknown.css", "@unknown-thing { .a { color: red; } }\n")
        inner = next(n for n in result.nodes if n.name == ".a")
        assert inner.scope_path == ("@unknown-thing",)

    def test_charset_and_import_and_supports(self, parser):
        result = parser.parse(
            "css/at.css",
            '@charset "utf-8";\n@import url("x.css");\n@supports (display: grid) { body { color: red; } }\n',
        )
        names = {n.name for n in result.nodes if n.kind == Kind.SECTION}
        assert {"@charset", "@import", "@supports"} <= names


class TestEdgeCases:
    def test_empty_file(self, parser):
        result = parser.parse("css/empty.css", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_rule_with_no_selectors_capture_is_skipped(self, parser):
        assert parser.process_definition_match({"rule": [object()]}, b"") is None


class TestDefensiveBranches:
    def test_process_definition_match_without_recognized_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_label_returns_none_for_rule_set_with_no_selectors_child(self):
        from indexter.parse.css import _label

        class FakeBlock:
            type = "block"

        class FakeRuleSet:
            type = "rule_set"
            children = [FakeBlock()]

        assert _label(FakeRuleSet()) is None

    def test_label_returns_none_for_unrecognized_node_type(self):
        from indexter.parse.css import _label

        class FakeNode:
            type = "stylesheet"
            children = []

        assert _label(FakeNode()) is None

    def test_label_for_keyframes_statement_ancestor(self):
        # tree-sitter-css never nests a `rule_set` inside a keyframe block in
        # practice, so this scope-segment branch is unreachable from real
        # source -- exercised directly.
        from indexter.parse.css import _label

        class FakeKeyframes:
            type = "keyframes_statement"
            children = []

        assert _label(FakeKeyframes()) == "@keyframes"

    def test_generic_at_rule_without_keyword_falls_back(self):
        from indexter.parse.css import _label

        class FakeChild:
            type = "block"
            text = None

        class FakeAtRule:
            type = "at_rule"
            children = [FakeChild()]

        assert _label(FakeAtRule()) == "@rule"
