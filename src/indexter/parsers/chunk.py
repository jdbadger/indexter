from collections.abc import Generator
from typing import Any

from .base import BaseParser


class ChunkParser(BaseParser):
    """Parser that splits source text into fixed-size chunks."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 25) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, content: str) -> Generator[tuple[str, dict[str, Any]]]:
        """Yield fixed-size chunks from the source text"""
        start = 0
        source_length = len(content)

        while start < source_length:
            end = min(start + self.chunk_size, source_length)
            chunk_content = content[start:end]

            yield (
                chunk_content,
                {
                    "language": "chunk",
                    "node_type": "chunk",
                    "node_name": None,
                    "start_byte": start,
                    "end_byte": end,
                    "start_line": content.count("\n", 0, start) + 1,
                    "end_line": content.count("\n", 0, end) + 1,
                    "documentation": None,
                    "parent_scope": None,
                    "signature": None,
                    "extra": {
                        "capture_name": "chunk",
                        "tree_sitter_type": "chunk",
                    },
                },
            )

            # Ensure we always advance by at least 1 to avoid infinite loop
            # when chunk_overlap >= chunk_size
            stride = max(1, self.chunk_size - self.chunk_overlap)
            next_start = start + stride
            if next_start >= source_length:
                break
            start = next_start
