from pathlib import Path

import pytest

from indexter.parse.json import JsonParser
from indexter.parse.models import Kind

FIXTURES = Path(__file__).parent / "fixtures" / "json"


@pytest.fixture
def parser():
    return JsonParser()


class TestNodes:
    def test_sample_fixture_nodes(self, parser):
        content = (FIXTURES / "sample.json").read_text()
        result = parser.parse("json/sample.json", content)
        data_nodes = [n for n in result.nodes if n.kind == Kind.DATA]
        by_name_scope = {(n.name, n.scope_path) for n in data_nodes}
        assert ("", ()) in by_name_scope  # root object
        assert ("settings", ()) in by_name_scope
        assert ("limits", ("settings",)) in by_name_scope

    def test_all_nodes_are_data_or_file(self, parser):
        content = (FIXTURES / "sample.json").read_text()
        result = parser.parse("json/sample.json", content)
        assert {n.kind for n in result.nodes} <= {Kind.FILE, Kind.DATA}

    def test_no_references(self, parser):
        content = (FIXTURES / "sample.json").read_text()
        result = parser.parse("json/sample.json", content)
        assert result.refs == []

    def test_array_index_scoping(self, parser):
        result = parser.parse("json/arr.json", '{"items": [{"a": 1}, {"b": [1, 2]}]}')
        names_scopes = {(n.name, n.scope_path) for n in result.nodes if n.kind == Kind.DATA}
        assert ("[0]", ("items",)) in names_scopes
        assert ("[1]", ("items",)) in names_scopes
        assert ("b", ("items", "[1]")) in names_scopes


class TestEdgeCases:
    def test_empty_file(self, parser):
        result = parser.parse("json/empty.json", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_scalar_only_document_has_no_data_nodes(self, parser):
        result = parser.parse("json/scalar.json", "42")
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_malformed_json_is_skipped(self, parser):
        result = parser.parse("json/broken.json", '{"a": }')
        assert all(n.kind != Kind.DATA for n in result.nodes)


class TestDefensiveBranches:
    def test_process_definition_match_without_def_capture(self, parser):
        assert parser.process_definition_match({}, b"") is None

    def test_array_index_falls_back_to_zero_when_target_not_found(self):
        from indexter.parse.json import _array_index

        class FakeChild:
            type = "number"

        class FakeArray:
            type = "array"
            children = [FakeChild()]

        class FakeTarget:
            parent = None

        assert _array_index(FakeArray(), FakeTarget()) == 0
