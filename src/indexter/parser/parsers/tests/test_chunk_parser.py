"""Tests for ChunkParser class."""

from indexter.parser.models import NodeMetadata
from indexter.parser.parsers.base import LanguageEnum
from indexter.parser.parsers.chunk import ChunkParser
from indexter.walker.models import Document, DocumentMetadata

# Unit Tests - ChunkParser.__init__


def test_chunk_parser_init_with_defaults(sample_document):
    """Test ChunkParser initialization with default parameters."""
    parser = ChunkParser()

    assert parser.chunk_size == 250
    assert parser.chunk_overlap == 25


def test_chunk_parser_init_with_custom_chunk_size(sample_document):
    """Test ChunkParser initialization with custom chunk_size."""
    parser = ChunkParser(chunk_size=500)

    assert parser.chunk_size == 500
    assert parser.chunk_overlap == 25


def test_chunk_parser_init_with_custom_overlap(sample_document):
    """Test ChunkParser initialization with custom chunk_overlap."""
    parser = ChunkParser(chunk_overlap=50)

    assert parser.chunk_size == 250
    assert parser.chunk_overlap == 50


def test_chunk_parser_init_with_custom_params(sample_document):
    """Test ChunkParser initialization with all custom parameters."""
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)

    assert parser.chunk_size == 100
    assert parser.chunk_overlap == 10


def test_chunk_parser_init_with_zero_overlap(sample_document):
    """Test ChunkParser initialization with zero overlap."""
    parser = ChunkParser(chunk_size=100, chunk_overlap=0)

    assert parser.chunk_overlap == 0


def test_chunk_parser_init_with_large_overlap(sample_document):
    """Test ChunkParser initialization with overlap >= chunk_size."""
    # This should not raise an error during init
    parser = ChunkParser(chunk_size=100, chunk_overlap=100)

    assert parser.chunk_overlap == 100


# Unit Tests - ChunkParser.parse


def test_chunk_parser_parse_returns_generator(sample_document):
    """Test that parse returns a generator."""
    parser = ChunkParser()
    result = parser.parse(sample_document)

    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_chunk_parser_parse_yields_tuples(sample_document):
    """Test that parse yields tuples of (content, metadata)."""
    parser = ChunkParser()

    for chunk_content, metadata in parser.parse(sample_document):
        assert isinstance(chunk_content, str)
        assert isinstance(metadata, NodeMetadata)


def test_chunk_parser_parse_chunk_content_from_document():
    """Test that chunks contain actual content from document."""
    doc = Document(
        path="test.txt",
        content="Hello World",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=11,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=5, chunk_overlap=0)

    chunks = list(parser.parse(doc))

    assert len(chunks) == 3
    assert chunks[0][0] == "Hello"
    assert chunks[1][0] == " Worl"
    assert chunks[2][0] == "d"


def test_chunk_parser_parse_respects_chunk_size():
    """Test that chunks respect the specified chunk_size."""
    content = "A" * 1000
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=1000,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=100, chunk_overlap=0)

    chunks = list(parser.parse(doc))

    # All chunks except possibly the last should be exactly chunk_size
    for _i, (chunk_content, _) in enumerate(chunks[:-1]):
        assert len(chunk_content) == 100

    # Last chunk can be <= chunk_size
    assert len(chunks[-1][0]) <= 100


def test_chunk_parser_parse_with_overlap():
    """Test that chunks overlap correctly."""
    content = "0123456789"  # 10 characters
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=10,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=5, chunk_overlap=2)

    chunks = list(parser.parse(doc))

    # chunk_size=5, overlap=2 means stride=3
    # Chunk 0: chars 0-4 (01234)
    # Chunk 1: chars 3-7 (34567)
    # Chunk 2: chars 6-9 (6789)
    assert chunks[0][0] == "01234"
    assert chunks[1][0] == "34567"
    assert chunks[2][0] == "6789"


def test_chunk_parser_parse_prevents_infinite_loop_when_overlap_equals_chunk_size():
    """Test that parser doesn't infinite loop when overlap >= chunk_size."""
    content = "A" * 100
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=100,
            mtime=123.0,
        ),
    )
    # When overlap == chunk_size, stride should be max(1, 0) = 1
    parser = ChunkParser(chunk_size=10, chunk_overlap=10)

    chunks = list(parser.parse(doc))

    # Should not hang, should produce chunks with stride of 1
    assert len(chunks) > 0
    assert len(chunks) <= 100  # At most one chunk per character


def test_chunk_parser_parse_prevents_infinite_loop_when_overlap_greater_than_chunk_size():
    """Test that parser doesn't infinite loop when overlap > chunk_size."""
    content = "A" * 100
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=100,
            mtime=123.0,
        ),
    )
    # When overlap > chunk_size, stride becomes negative, so max(1, negative) = 1
    parser = ChunkParser(chunk_size=10, chunk_overlap=20)

    chunks = list(parser.parse(doc))

    # Should not hang
    assert len(chunks) > 0
    assert len(chunks) <= 100


def test_chunk_parser_parse_empty_document():
    """Test parsing an empty document."""
    doc = Document(
        path="empty.txt",
        content="",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=0,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)

    chunks = list(parser.parse(doc))

    assert chunks == []


def test_chunk_parser_parse_single_character():
    """Test parsing a document with a single character."""
    doc = Document(
        path="single.txt",
        content="A",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=1,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)

    chunks = list(parser.parse(doc))

    assert len(chunks) == 1
    assert chunks[0][0] == "A"


def test_chunk_parser_parse_content_smaller_than_chunk_size():
    """Test parsing content smaller than chunk_size."""
    doc = Document(
        path="small.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)

    chunks = list(parser.parse(doc))

    assert len(chunks) == 1
    assert chunks[0][0] == "Hello"


def test_chunk_parser_parse_metadata_repo(sample_document):
    """Test that chunk metadata includes correct repo."""
    parser = ChunkParser()

    chunks = list(parser.parse(sample_document))

    for _, metadata in chunks:
        assert metadata.repo == sample_document.metadata.repo


def test_chunk_parser_parse_metadata_repo_path(sample_document):
    """Test that chunk metadata includes correct repo_path."""
    parser = ChunkParser()

    chunks = list(parser.parse(sample_document))

    for _, metadata in chunks:
        assert metadata.repo_path == sample_document.metadata.repo_path


def test_chunk_parser_parse_metadata_document_path(sample_document):
    """Test that chunk metadata includes correct document_path."""
    parser = ChunkParser()

    chunks = list(parser.parse(sample_document))

    for _, metadata in chunks:
        assert metadata.document_path == sample_document.path


def test_chunk_parser_parse_metadata_has_expected_fields(sample_document):
    """Test that chunk metadata includes expected fields."""
    parser = ChunkParser()

    chunks = list(parser.parse(sample_document))

    for _, metadata in chunks:
        # Verify metadata has expected fields
        assert metadata.repo == sample_document.metadata.repo
        assert metadata.repo_path == sample_document.metadata.repo_path


def test_chunk_parser_parse_metadata_language():
    """Test that chunk metadata has language set to N/A."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.language == LanguageEnum.NA
        assert metadata.language == "N/A"


def test_chunk_parser_parse_metadata_node_type():
    """Test that chunk metadata has node_type set to 'chunk'."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.node_type == "chunk"


def test_chunk_parser_parse_metadata_node_name():
    """Test that chunk metadata has node_name set to None."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.node_name is None


def test_chunk_parser_parse_metadata_start_and_end_bytes():
    """Test that chunk metadata has correct start_byte and end_byte."""
    content = "0123456789"
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=10,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=4, chunk_overlap=1)

    chunks = list(parser.parse(doc))

    # chunk_size=4, overlap=1 means stride=3
    # Chunk 0: bytes 0-3
    # Chunk 1: bytes 3-6
    # Chunk 2: bytes 6-9
    assert chunks[0][1].start_byte == 0
    assert chunks[0][1].end_byte == 4
    assert chunks[1][1].start_byte == 3
    assert chunks[1][1].end_byte == 7
    assert chunks[2][1].start_byte == 6
    assert chunks[2][1].end_byte == 10


def test_chunk_parser_parse_metadata_start_and_end_lines_single_line():
    """Test line numbers for single-line content."""
    doc = Document(
        path="test.txt",
        content="Hello World",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=11,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=5, chunk_overlap=0)

    chunks = list(parser.parse(doc))

    # All chunks on line 1 (1-indexed)
    for _, metadata in chunks:
        assert metadata.start_line == 1
        assert metadata.end_line == 1


def test_chunk_parser_parse_metadata_start_and_end_lines_multiline():
    """Test line numbers for multi-line content."""
    content = "line1\nline2\nline3"  # 17 chars
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=17,
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=10, chunk_overlap=0)

    chunks = list(parser.parse(doc))

    # Chunk 0: "line1\nline" (chars 0-9) - spans lines 1-2
    # Chunk 1: "2\nline3" (chars 10-16) - spans lines 2-3
    assert chunks[0][1].start_line == 1
    # end_line is 1 + count of newlines in chars 0-9
    # chars 0-9 = "line1\nline", has 1 newline, so end_line = 2
    assert chunks[0][1].end_line == 2

    assert chunks[1][1].start_line == 2
    # chars 10-16 = "2\nline3", has 1 newline at position 11
    # count("\n", 0, 16) = 2 newlines total, so end_line = 3
    assert chunks[1][1].end_line == 3


def test_chunk_parser_parse_metadata_documentation():
    """Test that chunk metadata has documentation set to None."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.documentation is None


def test_chunk_parser_parse_metadata_parent_scope():
    """Test that chunk metadata has parent_scope set to None."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.parent_scope is None


def test_chunk_parser_parse_metadata_signature():
    """Test that chunk metadata has signature set to None."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.signature is None


def test_chunk_parser_parse_metadata_extra():
    """Test that chunk metadata has correct extra fields."""
    doc = Document(
        path="test.txt",
        content="Hello",
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=5,
            mtime=123.0,
        ),
    )
    parser = ChunkParser()

    chunks = list(parser.parse(doc))

    for _, metadata in chunks:
        assert metadata.extra == {
            "capture_name": "chunk",
            "tree_sitter_type": "chunk",
        }


def test_chunk_parser_parse_unicode_content():
    """Test parsing content with unicode characters."""
    content = "Hello 世界 🌍"
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=len(content.encode()),
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=5, chunk_overlap=0)

    chunks = list(parser.parse(doc))

    # Should handle unicode correctly
    assert len(chunks) > 0
    # Reassemble chunks should give original content
    reassembled = "".join(chunk[0] for chunk in chunks)
    assert reassembled == content


def test_chunk_parser_parse_with_whitespace():
    """Test parsing content with various whitespace characters."""
    content = "Hello\n\tWorld\r\nTest"
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=len(content),
            mtime=123.0,
        ),
    )
    parser = ChunkParser(chunk_size=10, chunk_overlap=0)

    chunks = list(parser.parse(doc))

    # Should preserve whitespace
    reassembled = "".join(chunk[0] for chunk in chunks)
    assert reassembled == content


# Integration Tests


def test_chunk_parser_integration_realistic_python_file():
    """Integration test: parse a realistic Python file."""
    python_content = '''"""Module docstring."""

import os
import sys

def function_one():
    """Function docstring."""
    return "Hello"

def function_two():
    """Another function."""
    return "World"

class MyClass:
    """Class docstring."""
    
    def method(self):
        """Method docstring."""
        pass
'''
    doc = Document(
        path="module.py",
        content=python_content,
        metadata=DocumentMetadata(
            repo="my-repo",
            repo_path="/path/to/my-repo",
            ext=".py",
            size_bytes=len(python_content),
            mtime=1234567890.0,
        ),
    )

    parser = ChunkParser(chunk_size=100, chunk_overlap=20)
    chunks = list(parser.parse(doc))

    # Verify chunks were created
    assert len(chunks) > 0

    # Verify all chunks have correct metadata structure
    for chunk_content, metadata in chunks:
        assert isinstance(chunk_content, str)
        assert metadata.repo == "my-repo"
        assert metadata.repo_path == "/path/to/my-repo"
        assert metadata.document_path == "module.py"
        assert metadata.language == "N/A"
        assert metadata.node_type == "chunk"
        assert len(chunk_content) <= 100

    # Verify chunks overlap correctly (stride = 100 - 20 = 80)
    if len(chunks) > 1:
        # Second chunk should start 80 chars after first
        assert chunks[1][1].start_byte == 80


def test_chunk_parser_integration_markdown_document():
    """Integration test: parse a markdown document."""
    markdown_content = """# Title

## Section 1

This is a paragraph with some **bold** and *italic* text.

## Section 2

- Item 1
- Item 2
- Item 3

```python
def example():
    return "code block"
```

## Section 3

More content here.
"""
    doc = Document(
        path="README.md",
        content=markdown_content,
        metadata=DocumentMetadata(
            repo="docs-repo",
            repo_path="/docs",
            ext=".md",
            size_bytes=len(markdown_content),
            mtime=9999999.0,
        ),
    )

    parser = ChunkParser(chunk_size=150, chunk_overlap=30)
    chunks = list(parser.parse(doc))

    assert len(chunks) > 0

    # Verify metadata consistency across all chunks
    for _, metadata in chunks:
        assert metadata.repo == "docs-repo"
        assert metadata.document_path == "README.md"
        assert metadata.node_type == "chunk"


def test_chunk_parser_integration_large_file_performance():
    """Integration test: parse a large file efficiently."""
    # Create a large document (10,000 characters)
    large_content = "A" * 10000
    doc = Document(
        path="large.txt",
        content=large_content,
        metadata=DocumentMetadata(
            repo="perf-test",
            repo_path="/perf",
            ext=".txt",
            size_bytes=10000,
            mtime=123.0,
        ),
    )

    parser = ChunkParser(chunk_size=500, chunk_overlap=50)
    chunks = list(parser.parse(doc))

    # With stride = 500 - 50 = 450
    # Expected chunks = ceil(10000 / 450) ≈ 23 chunks
    expected_chunks = 23
    assert len(chunks) >= expected_chunks - 1  # Allow some variance
    assert len(chunks) <= expected_chunks + 1

    # Verify no duplicates in start positions (except for overlap)
    start_positions = [metadata.start_byte for _, metadata in chunks]
    # Each start should be unique
    assert len(set(start_positions)) == len(start_positions)


def test_chunk_parser_integration_no_overlap_coverage():
    """Integration test: verify complete coverage with no overlap."""
    content = "0123456789ABCDEFGHIJ"  # 20 characters
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=20,
            mtime=123.0,
        ),
    )

    parser = ChunkParser(chunk_size=5, chunk_overlap=0)
    chunks = list(parser.parse(doc))

    # Should have exactly 4 chunks with no gaps or overlaps
    assert len(chunks) == 4

    # Reassemble to verify complete coverage
    reassembled = "".join(chunk[0] for chunk in chunks)
    assert reassembled == content

    # Verify no gaps
    for i in range(len(chunks) - 1):
        # End of chunk i should equal start of chunk i+1
        assert chunks[i][1].end_byte == chunks[i + 1][1].start_byte


def test_chunk_parser_integration_with_overlap_coverage():
    """Integration test: verify complete coverage with overlap."""
    content = "ABCDEFGHIJ"  # 10 characters
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=10,
            mtime=123.0,
        ),
    )

    parser = ChunkParser(chunk_size=4, chunk_overlap=1)
    chunks = list(parser.parse(doc))

    # stride = 4 - 1 = 3
    # Chunks at: 0-3, 3-6, 6-9, 9-9 (last partial)
    assert len(chunks) == 4

    # Verify overlaps exist
    assert chunks[0][0][-1] == chunks[1][0][0]  # D overlaps
    assert chunks[1][0][-1] == chunks[2][0][0]  # G overlaps

    # Every character should appear in at least one chunk
    covered_chars = set()
    for chunk_content, metadata in chunks:
        for i, _char in enumerate(chunk_content):
            covered_chars.add(metadata.start_byte + i)

    # All positions 0-9 should be covered
    assert covered_chars == set(range(10))


def test_chunk_parser_integration_boundary_conditions():
    """Integration test: test various boundary conditions."""
    # Test exact multiple of chunk_size
    content = "A" * 100
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=100,
            mtime=123.0,
        ),
    )

    parser = ChunkParser(chunk_size=50, chunk_overlap=0)
    chunks = list(parser.parse(doc))

    # Should have exactly 2 chunks
    assert len(chunks) == 2
    assert len(chunks[0][0]) == 50
    assert len(chunks[1][0]) == 50


def test_chunk_parser_integration_very_small_chunks():
    """Integration test: parse with very small chunk size."""
    content = "Hello World!"
    doc = Document(
        path="test.txt",
        content=content,
        metadata=DocumentMetadata(
            repo="test",
            repo_path="/test",
            ext=".txt",
            size_bytes=len(content),
            mtime=123.0,
        ),
    )

    parser = ChunkParser(chunk_size=1, chunk_overlap=0)
    chunks = list(parser.parse(doc))

    # Should have one chunk per character
    assert len(chunks) == len(content)

    # Each chunk should be a single character
    for i, (chunk_content, _) in enumerate(chunks):
        assert chunk_content == content[i]


def test_chunk_parser_integration_json_document():
    """Integration test: parse a JSON document."""
    json_content = """{
    "name": "test",
    "version": "1.0.0",
    "dependencies": {
        "package1": "^1.0.0",
        "package2": "~2.3.4"
    },
    "scripts": {
        "test": "pytest",
        "build": "python setup.py build"
    }
}"""

    doc = Document(
        path="package.json",
        content=json_content,
        metadata=DocumentMetadata(
            repo="json-repo",
            repo_path="/json",
            ext=".json",
            size_bytes=len(json_content),
            mtime=555555.0,
        ),
    )

    parser = ChunkParser(chunk_size=80, chunk_overlap=15)
    chunks = list(parser.parse(doc))

    assert len(chunks) > 0

    # Verify structure preserved
    for _chunk_content, metadata in chunks:
        assert metadata.document_path == "package.json"
        assert metadata.language == "N/A"
