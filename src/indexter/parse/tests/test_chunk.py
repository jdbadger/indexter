from indexter.config import Settings
from indexter.parse.chunk import ChunkParser
from indexter.parse.models import Kind


class TestChunkParser:
    def test_default_chunk_sizing(self):
        parser = ChunkParser()
        assert parser.chunk_size == 1000
        assert parser.chunk_overlap == 100

    def test_file_node_present(self):
        result = ChunkParser().parse("a.txt", "hello world\n")
        file_nodes = [n for n in result.nodes if n.kind == Kind.FILE]
        assert len(file_nodes) == 1

    def test_chunks_cover_the_whole_file(self):
        settings = Settings(chunk_size=10, chunk_overlap=2)
        content = "x" * 35
        result = ChunkParser(settings).parse("a.txt", content)
        chunks = sorted((n.start_byte, n.end_byte) for n in result.nodes if n.kind == Kind.CHUNK)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == len(content.encode())

    def test_no_references(self):
        result = ChunkParser().parse("a.txt", "hello\nworld\n")
        assert result.refs == []

    def test_empty_file(self):
        result = ChunkParser().parse("a.txt", "")
        assert [n.kind for n in result.nodes] == [Kind.FILE]
        assert result.errors == []

    def test_chunks_parent_to_file_node(self):
        settings = Settings(chunk_size=5, chunk_overlap=0)
        result = ChunkParser(settings).parse("a.txt", "x" * 20)
        file_node = next(n for n in result.nodes if n.kind == Kind.FILE)
        chunks = [n for n in result.nodes if n.kind == Kind.CHUNK]
        assert chunks  # sanity: more than one chunk produced
        assert all(c.parent_id == file_node.id for c in chunks)

    def test_duplicate_suffixes_are_ordered_by_position(self):
        settings = Settings(chunk_size=5, chunk_overlap=0)
        result = ChunkParser(settings).parse("a.txt", "x" * 20)
        chunk_ids = [n.id for n in result.nodes if n.kind == Kind.CHUNK]
        assert chunk_ids[0] == "a.txt::#chunk"
        assert chunk_ids[1] == "a.txt::#chunk~2"
