import dataclasses

import pytest

from indexter.parse.models import Kind, ParsedNode, ParsedRef, ParseResult, RawRef, RefKind


class TestKind:
    def test_membership_is_closed(self):
        assert {k.value for k in Kind} == {
            "file",
            "class",
            "function",
            "method",
            "constant",
            "interface",
            "type_alias",
            "enum",
            "struct",
            "trait",
            "section",
            "data",
            "chunk",
            "external_module",
        }


class TestRefKind:
    def test_membership_is_closed(self):
        assert {k.value for k in RefKind} == {"calls", "imports", "inherits"}


def _node(**overrides):
    defaults = dict(
        kind=Kind.FUNCTION,
        name="foo",
        scope_path=(),
        language="python",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=10,
    )
    defaults.update(overrides)
    return ParsedNode(**defaults)


class TestParsedNode:
    def test_is_frozen(self):
        node = _node()
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.name = "bar"

    def test_id_and_parent_default_to_placeholders(self):
        node = _node()
        assert node.id == ""
        assert node.parent_id is None

    def test_optional_fields_default_to_none(self):
        node = _node()
        assert node.signature is None
        assert node.docstring is None


class TestParsedRef:
    def test_is_frozen(self):
        ref = ParsedRef(
            from_node_id="a.py::foo#function", raw_name="bar", head="bar", ref_kind=RefKind.CALLS, line=1, col=1
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.raw_name = "baz"


class TestRawRef:
    def test_is_frozen(self):
        raw = RawRef(origin_byte=0, raw_name="bar", head="bar", ref_kind=RefKind.CALLS, line=1, col=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            raw.raw_name = "baz"


class TestParseResult:
    def test_defaults_to_empty_collections(self):
        result = ParseResult()
        assert result.nodes == []
        assert result.refs == []
        assert result.errors == []

    def test_defaults_are_independent_across_instances(self):
        a = ParseResult()
        b = ParseResult()
        a.nodes.append(_node())
        assert b.nodes == []
