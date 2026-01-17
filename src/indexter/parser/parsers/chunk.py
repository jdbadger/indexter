from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indexter.models import Document

from .base import BaseParser, LanguageEnum, NodeMetadata


class ChunkParser(BaseParser):
    """Parser that splits source text into fixed-size chunks."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 25) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, document: Document) -> Generator[tuple[str, NodeMetadata]]:
        """Yield fixed-size chunks from the source text"""
        content = document.content
        start = 0
        source_length = len(content)

        while start < source_length:
            end = min(start + self.chunk_size, source_length)
            chunk_content = content[start:end]

            yield (
                chunk_content,
                NodeMetadata(
                    repo=document.metadata.repo,
                    repo_path=document.metadata.repo_path,
                    document_path=document.path,
                    hash=document.metadata.hash,
                    language=LanguageEnum.NA,
                    node_type="chunk",
                    node_name=None,
                    start_byte=start,
                    end_byte=end,
                    start_line=content.count("\n", 0, start) + 1,
                    end_line=content.count("\n", 0, end) + 1,
                    documentation=None,
                    parent_scope=None,
                    signature=None,
                    extra={
                        "capture_name": "chunk",
                        "tree_sitter_type": "chunk",
                    },
                ),
            )

            # Ensure we always advance by at least 1 to avoid infinite loop
            # when chunk_overlap >= chunk_size
            stride = max(1, self.chunk_size - self.chunk_overlap)
            next_start = start + stride
            if next_start >= source_length:
                break
            start = next_start
