from pathlib import Path

import pytest

from indexter.parse.models import Kind
from indexter.parse.toml import TomlParser

FIXTURES = Path(__file__).parent / "fixtures" / "toml"


@pytest.fixture
def parser():
    return TomlParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.toml").read_text()
        result = parser.parse("toml/sample.toml", content)
        by_name_scope = {(n.name, n.scope_path) for n in result.nodes if n.kind == Kind.DATA}
        assert ("name", ()) in by_name_scope
        assert ("version", ()) in by_name_scope
        assert ("settings", ()) in by_name_scope

    def test_duplicate_table_array_elements_are_suffixed(self, parser):
        content = (FIXTURES / "sample.toml").read_text()
        result = parser.parse("toml/sample.toml", content)
        items_ids = sorted(n.id for n in result.nodes if n.name == "items")
        assert items_ids == ["toml/sample.toml::items#data", "toml/sample.toml::items#data~2"]

    def test_all_nodes_are_data_or_file(self, parser):
        content = (FIXTURES / "sample.toml").read_text()
        result = parser.parse("toml/sample.toml", content)
        assert {n.kind for n in result.nodes} <= {Kind.FILE, Kind.DATA}

    def test_no_references(self, parser):
        content = (FIXTURES / "sample.toml").read_text()
        result = parser.parse("toml/sample.toml", content)
        assert result.refs == []

    def test_dotted_table_name_splits_into_scope_path(self, parser):
        result = parser.parse("toml/dotted.toml", "[a.b.c]\nx = 1\n")
        node = next(n for n in result.nodes if n.kind == Kind.DATA and n.name == "c")
        assert node.scope_path == ("a", "b")


class TestEdgeCases:
    def test_empty_file(self, parser):
        result = parser.parse("toml/empty.toml", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]


class TestDefensiveBranches:
    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_missing_key_child_is_skipped(self, parser):
        class FakePair:
            type = "pair"
            has_error = False
            children = []

        assert parser.process_definition_match({"def": [FakePair()]}, b"") is None

    def test_error_node_is_skipped(self, parser):
        class FakePair:
            type = "pair"
            has_error = True
            children = []

        assert parser.process_definition_match({"def": [FakePair()]}, b"") is None
