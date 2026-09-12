"""Cross-cutting checks for every non-code fixture (task 7.6): every node's
kind is in the closed vocabulary, no parser produces a reference, and the
chunk fallback covers the whole file for an unregistered extension.
"""

from pathlib import Path

import pytest

from indexter.parse.base import parse_file
from indexter.parse.models import Kind

FIXTURES = Path(__file__).parent / "fixtures"

NON_CODE_FIXTURES = [
    "markdown/sample.md",
    "json/sample.json",
    "yaml/sample.yaml",
    "toml/sample.toml",
    "html/sample.html",
    "css/sample.css",
]


@pytest.mark.parametrize("relpath", NON_CODE_FIXTURES)
def test_every_node_kind_is_in_the_closed_vocabulary(relpath):
    content = (FIXTURES / relpath).read_text()
    result = parse_file(relpath, content)
    assert result.errors == []
    assert result.nodes
    assert all(isinstance(n.kind, Kind) for n in result.nodes)


@pytest.mark.parametrize("relpath", NON_CODE_FIXTURES)
def test_no_references_are_produced(relpath):
    content = (FIXTURES / relpath).read_text()
    result = parse_file(relpath, content)
    assert result.refs == []


def test_unregistered_extension_falls_back_to_chunk_covering_whole_file():
    relpath = "other/sample.xyz"
    content = (FIXTURES / relpath).read_text()
    result = parse_file(relpath, content)
    length = len(content.encode())
    chunks = sorted((n.start_byte, n.end_byte) for n in result.nodes if n.kind == Kind.CHUNK)
    assert chunks
    assert chunks[0][0] == 0
    assert chunks[-1][1] == length
    assert result.refs == []
