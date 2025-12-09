"""Tests for ChunkParser."""

from indexter.parsers.chunk import ChunkParser


def test_chunk_parser_initialization_default():
    """Test ChunkParser initializes with default values."""
    parser = ChunkParser()
    assert parser.chunk_size == 250
    assert parser.chunk_overlap == 25


def test_chunk_parser_initialization_custom():
    """Test ChunkParser initializes with custom values."""
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)
    assert parser.chunk_size == 100
    assert parser.chunk_overlap == 10


def test_chunk_parser_empty_string():
    """Test parsing empty string yields nothing."""
    parser = ChunkParser()
    result = list(parser.parse(""))
    assert result == []


def test_chunk_parser_single_small_chunk():
    """Test parsing text smaller than chunk size yields single chunk."""
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)
    source = "Hello, World!"

    result = list(parser.parse(source))

    assert len(result) == 1
    content, metadata = result[0]

    assert content == "Hello, World!"
    assert metadata["language"] == "chunk"
    assert metadata["node_type"] == "chunk"
    assert metadata["node_name"] is None
    assert metadata["start_byte"] == 0
    assert metadata["end_byte"] == 13
    assert metadata["start_line"] == 1
    assert metadata["end_line"] == 1
    assert metadata["documentation"] is None
    assert metadata["parent_scope"] is None
    assert metadata["signature"] is None
    assert metadata["extra"]["capture_name"] == "chunk"
    assert metadata["extra"]["tree_sitter_type"] == "chunk"


def test_chunk_parser_exact_chunk_size():
    """Test parsing text exactly equal to chunk size."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=0)
    source = "0123456789"

    result = list(parser.parse(source))

    assert len(result) == 1
    content, metadata = result[0]
    assert content == "0123456789"
    assert metadata["start_byte"] == 0
    assert metadata["end_byte"] == 10


def test_chunk_parser_multiple_chunks_no_overlap():
    """Test parsing text into multiple chunks with no overlap."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=0)
    source = "0123456789ABCDEFGHIJ"

    result = list(parser.parse(source))

    assert len(result) == 2

    content1, metadata1 = result[0]
    assert content1 == "0123456789"
    assert metadata1["start_byte"] == 0
    assert metadata1["end_byte"] == 10

    content2, metadata2 = result[1]
    assert content2 == "ABCDEFGHIJ"
    assert metadata2["start_byte"] == 10
    assert metadata2["end_byte"] == 20


def test_chunk_parser_multiple_chunks_with_overlap():
    """Test parsing text into multiple chunks with overlap."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=3)
    source = "0123456789ABCDEFGHIJ"

    result = list(parser.parse(source))

    assert len(result) == 3

    content1, metadata1 = result[0]
    assert content1 == "0123456789"
    assert metadata1["start_byte"] == 0
    assert metadata1["end_byte"] == 10

    # Next chunk starts at 10 - 3 = 7
    content2, metadata2 = result[1]
    assert content2 == "789ABCDEFG"
    assert metadata2["start_byte"] == 7
    assert metadata2["end_byte"] == 17

    # Next chunk starts at 7 + 7 = 14
    content3, metadata3 = result[2]
    assert content3 == "EFGHIJ"
    assert metadata3["start_byte"] == 14
    assert metadata3["end_byte"] == 20


def test_chunk_parser_line_counting_single_line():
    """Test line counting for single line text."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=0)
    source = "Single line text"

    result = list(parser.parse(source))

    content, metadata = result[0]
    assert metadata["start_line"] == 1
    assert metadata["end_line"] == 1


def test_chunk_parser_line_counting_multiple_lines():
    """Test line counting for multiline text."""
    parser = ChunkParser(chunk_size=20, chunk_overlap=0)
    source = "Line 1\nLine 2\nLine 3\nLine 4"

    result = list(parser.parse(source))

    assert len(result) == 2

    content1, metadata1 = result[0]
    assert content1 == "Line 1\nLine 2\nLine 3"
    assert metadata1["start_line"] == 1
    assert metadata1["end_line"] == 3

    content2, metadata2 = result[1]
    assert content2 == "\nLine 4"
    assert metadata2["start_line"] == 3
    assert metadata2["end_line"] == 4


def test_chunk_parser_line_counting_with_overlap():
    """Test line counting works correctly with overlapping chunks."""
    parser = ChunkParser(chunk_size=15, chunk_overlap=5)
    source = "Line 1\nLine 2\nLine 3\nLine 4"

    result = list(parser.parse(source))

    # First chunk: "Line 1\nLine 2\n"
    content1, metadata1 = result[0]
    assert metadata1["start_line"] == 1
    assert metadata1["end_line"] == 3  # Ends at start of Line 3

    # Second chunk starts at byte 10: "ine 2\nLine 3\nL"
    content2, metadata2 = result[1]
    assert metadata2["start_line"] == 2  # 1 newline before start
    assert metadata2["end_line"] == 4


def test_chunk_parser_last_chunk_smaller():
    """Test that last chunk can be smaller than chunk_size."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=0)
    source = "0123456789ABC"

    result = list(parser.parse(source))

    assert len(result) == 2
    content2, metadata2 = result[1]
    assert content2 == "ABC"
    assert len(content2) == 3


def test_chunk_parser_overlap_larger_than_chunk():
    """Test behavior when overlap is larger than chunk size."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=15)
    source = "0123456789ABCDEFGHIJ"

    result = list(parser.parse(source))

    # Should still work, just creates overlapping chunks
    assert len(result) >= 1
    # First chunk should still be normal
    content1, metadata1 = result[0]
    assert content1 == "0123456789"


def test_chunk_parser_unicode_text():
    """Test parsing Unicode text."""
    parser = ChunkParser(chunk_size=20, chunk_overlap=5)
    source = "Hello 世界 🌍 Test"

    result = list(parser.parse(source))

    assert len(result) >= 1
    content1, metadata1 = result[0]
    assert "Hello" in content1
    assert metadata1["start_byte"] == 0


def test_chunk_parser_whitespace_only():
    """Test parsing whitespace-only text."""
    parser = ChunkParser(chunk_size=10, chunk_overlap=2)
    source = "          "  # 10 spaces

    result = list(parser.parse(source))

    # With chunk_size=10 and overlap=2, stride is 8
    # First chunk: bytes 0-10, second starts at 8 but 8 < 10 so we get a second chunk
    assert len(result) == 2
    content, metadata = result[0]
    assert content == "          "
    assert metadata["start_byte"] == 0
    assert metadata["end_byte"] == 10


def test_chunk_parser_newlines_only():
    """Test parsing text with only newlines."""
    parser = ChunkParser(chunk_size=5, chunk_overlap=0)
    source = "\n\n\n\n\n\n\n\n\n\n"  # 10 newlines

    result = list(parser.parse(source))

    assert len(result) == 2
    content1, metadata1 = result[0]
    assert content1 == "\n\n\n\n\n"
    assert metadata1["start_line"] == 1
    assert metadata1["end_line"] == 6  # 5 newlines means we're at line 6


def test_chunk_parser_generator_behavior():
    """Test that parse returns a generator."""
    parser = ChunkParser()
    source = "Test text"

    result = parser.parse(source)

    # Should be a generator
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_chunk_parser_large_text():
    """Test parsing large text generates expected number of chunks."""
    parser = ChunkParser(chunk_size=100, chunk_overlap=10)
    # Create text of 500 characters
    source = "x" * 500

    result = list(parser.parse(source))

    # With chunk_size=100 and overlap=10, stride is 90
    # Expected chunks: starts at 0, 90, 180, 270, 360, 450
    # That's 6 chunks
    assert len(result) == 6

    # Verify first and last chunks
    assert result[0][1]["start_byte"] == 0
    assert result[0][1]["end_byte"] == 100
    assert result[-1][1]["start_byte"] == 450
    assert result[-1][1]["end_byte"] == 500


def test_chunk_parser_metadata_structure():
    """Test that all metadata fields are present and have correct types."""
    parser = ChunkParser()
    source = "Test"

    result = list(parser.parse(source))
    content, metadata = result[0]

    # Check all required fields exist
    assert "language" in metadata
    assert "node_type" in metadata
    assert "node_name" in metadata
    assert "start_byte" in metadata
    assert "end_byte" in metadata
    assert "start_line" in metadata
    assert "end_line" in metadata
    assert "documentation" in metadata
    assert "parent_scope" in metadata
    assert "signature" in metadata
    assert "extra" in metadata

    # Check types
    assert isinstance(metadata["language"], str)
    assert isinstance(metadata["node_type"], str)
    assert isinstance(metadata["start_byte"], int)
    assert isinstance(metadata["end_byte"], int)
    assert isinstance(metadata["start_line"], int)
    assert isinstance(metadata["end_line"], int)
    assert isinstance(metadata["extra"], dict)


def test_chunk_parser_chunk_size_one():
    """Test edge case with chunk_size=1."""
    parser = ChunkParser(chunk_size=1, chunk_overlap=0)
    source = "ABC"

    result = list(parser.parse(source))

    assert len(result) == 3
    assert result[0][0] == "A"
    assert result[1][0] == "B"
    assert result[2][0] == "C"


def test_chunk_parser_zero_overlap():
    """Test that zero overlap works correctly."""
    parser = ChunkParser(chunk_size=5, chunk_overlap=0)
    source = "0123456789"

    result = list(parser.parse(source))

    assert len(result) == 2
    assert result[0][0] == "01234"
    assert result[1][0] == "56789"
    # Chunks should not overlap
    assert result[0][1]["end_byte"] == result[1][1]["start_byte"]
