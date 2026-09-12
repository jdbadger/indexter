from pathlib import Path

import pytest

from indexter.parse.models import Kind
from indexter.parse.yaml import YamlParser

FIXTURES = Path(__file__).parent / "fixtures" / "yaml"


@pytest.fixture
def parser():
    return YamlParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.yaml").read_text()
        result = parser.parse("yaml/sample.yaml", content)
        by_name_scope = {(n.name, n.scope_path) for n in result.nodes if n.kind == Kind.DATA}
        assert ("", ()) in by_name_scope  # root mapping
        assert ("settings", ()) in by_name_scope
        assert ("limits", ("settings",)) in by_name_scope

    def test_all_nodes_are_data_or_file(self, parser):
        content = (FIXTURES / "sample.yaml").read_text()
        result = parser.parse("yaml/sample.yaml", content)
        assert {n.kind for n in result.nodes} <= {Kind.FILE, Kind.DATA}

    def test_no_references(self, parser):
        content = (FIXTURES / "sample.yaml").read_text()
        result = parser.parse("yaml/sample.yaml", content)
        assert result.refs == []

    def test_sequence_index_scoping(self, parser):
        result = parser.parse("yaml/seq.yaml", "items:\n  - a: 1\n  - b:\n      - 1\n      - 2\n")
        names_scopes = {(n.name, n.scope_path) for n in result.nodes if n.kind == Kind.DATA}
        assert ("[1]", ("items",)) in names_scopes
        assert ("b", ("items", "[1]")) in names_scopes


class TestEdgeCases:
    def test_empty_file(self, parser):
        result = parser.parse("yaml/empty.yaml", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_scalar_only_document_has_no_data_nodes(self, parser):
        result = parser.parse("yaml/scalar.yaml", "just a string\n")
        assert [n.kind for n in result.nodes] == [Kind.FILE]


class TestDefensiveBranches:
    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_error_node_is_skipped(self, parser):
        class FakeMapping:
            type = "block_mapping"
            has_error = True
            children = []

        assert parser.process_definition_match({"def": [FakeMapping()]}, b"") is None

    def test_sequence_index_skips_non_item_children(self):
        from indexter.parse.yaml import _sequence_index

        class FakeComment:
            type = "comment"

        class FakeSequence:
            children = [FakeComment()]

        class FakeTarget:
            parent = None

        assert _sequence_index(FakeSequence(), FakeTarget()) == 0

    def test_key_text_falls_back_to_whole_node_text(self):
        from indexter.parse.yaml import _key_text

        class FakeChild:
            type = "flow_node"
            text = None
            children = []

        class FakeKey:
            type = "flow_node"
            text = b"raw"
            children = [FakeChild()]

        assert _key_text(FakeKey()) == "raw"
