"""Fallback parser for files with no registered language: fixed-size,
overlapping byte-range chunks. No references -- there is no syntax to
extract them from.

Chunks by UTF-8 bytes rather than characters (the lifted implementation
chunked by character), so `start_byte`/`end_byte` are directly usable byte
offsets like every other parser's, consistent with the schema's byte-range
columns. Defaults (1000/100 bytes) are sized for the composer's ~256-token
embedding budget at roughly 3.5 bytes per token.
"""

from __future__ import annotations

from indexter.config import Settings
from indexter.parse.base import BaseParser
from indexter.parse.ids import assign_ids, link_parents
from indexter.parse.models import Kind, ParsedNode, ParseResult


class ChunkParser(BaseParser):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings if settings is not None else Settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def parse(self, relpath: str, content: str) -> ParseResult:
        source_bytes = content.encode("utf-8")
        length = len(source_bytes)

        def line_at(byte_offset: int) -> int:
            return source_bytes.count(b"\n", 0, byte_offset) + 1

        nodes: list[ParsedNode] = [
            ParsedNode(
                kind=Kind.FILE,
                name="",
                scope_path=(),
                language="",
                start_line=1,
                end_line=line_at(length),
                start_byte=0,
                end_byte=length,
            )
        ]

        stride = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < length:
            end = min(start + self.chunk_size, length)
            nodes.append(
                ParsedNode(
                    kind=Kind.CHUNK,
                    name="",
                    scope_path=(),
                    language="",
                    start_line=line_at(start),
                    end_line=line_at(end),
                    start_byte=start,
                    end_byte=end,
                )
            )
            next_start = start + stride
            if next_start >= length:
                break
            start = next_start

        with_ids = assign_ids(relpath, nodes)
        linked = link_parents(with_ids)
        return ParseResult(nodes=linked, refs=[], errors=[])
