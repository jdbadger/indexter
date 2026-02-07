"""Tests for parser models."""

import uuid

import pytest
from pydantic import ValidationError

from indexter.parser.models import Node, NodeMetadata


class TestNodeMetadata:
    """Tests for NodeMetadata model."""

    def test_should_create_metadata_with_valid_data(self, valid_node_metadata_data):
        """Test NodeMetadata creation with all valid fields."""
        metadata = NodeMetadata(**valid_node_metadata_data)

        assert metadata.repo == "test-repo"
        assert metadata.repo_path == "/home/user/repos/test-repo"
        assert metadata.document_path == "src/main.py"
        assert metadata.document_hash == "abc123def456"
        assert metadata.language == "python"
        assert metadata.node_type == "function"
        assert metadata.node_name == "my_function"
        assert metadata.start_byte == 0
        assert metadata.end_byte == 100
        assert metadata.start_line == 1
        assert metadata.end_line == 10
        assert metadata.documentation == "This is a docstring."
        assert metadata.parent_scope is None
        assert metadata.signature == "def my_function(arg1: str, arg2: int) -> bool"
        assert metadata.extra == {}

    @pytest.mark.parametrize(
        "field_name",
        [
            "repo",
            "repo_path",
            "document_path",
            "document_hash",
            "language",
            "node_type",
            "start_byte",
            "end_byte",
            "start_line",
            "end_line",
        ],
    )
    def test_should_reject_missing_required_field(self, valid_node_metadata_data, field_name):
        """Test NodeMetadata rejects missing required fields."""
        data = valid_node_metadata_data.copy()
        del data[field_name]

        with pytest.raises(ValidationError) as exc_info:
            NodeMetadata(**data)

        assert field_name in str(exc_info.value)

    def test_should_accept_optional_fields_as_none(self, valid_node_metadata_data):
        """Test NodeMetadata accepts None for optional fields."""
        data = valid_node_metadata_data.copy()
        data["node_name"] = None
        data["documentation"] = None
        data["parent_scope"] = None
        data["signature"] = None

        metadata = NodeMetadata(**data)

        assert metadata.node_name is None
        assert metadata.documentation is None
        assert metadata.parent_scope is None
        assert metadata.signature is None

    def test_should_use_default_empty_dict_for_extra(self, valid_node_metadata_data):
        """Test NodeMetadata uses empty dict as default for extra."""
        data = valid_node_metadata_data.copy()
        del data["extra"]

        metadata = NodeMetadata(**data)

        assert metadata.extra == {}

    def test_should_serialize_to_dict(self, sample_node_metadata):
        """Test NodeMetadata can be serialized to dict."""
        result = sample_node_metadata.model_dump()

        assert isinstance(result, dict)
        assert result["repo"] == "test-repo"
        assert result["language"] == "python"

    def test_should_serialize_to_json(self, sample_node_metadata):
        """Test NodeMetadata can be serialized to JSON."""
        json_str = sample_node_metadata.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-repo" in json_str
        assert "python" in json_str

    def test_should_deserialize_from_dict(self, valid_node_metadata_data):
        """Test NodeMetadata can be deserialized from dict."""
        metadata = NodeMetadata.model_validate(valid_node_metadata_data)

        assert metadata.repo == valid_node_metadata_data["repo"]
        assert metadata.language == valid_node_metadata_data["language"]

    def test_should_support_field_descriptions(self):
        """Test NodeMetadata fields have descriptions."""
        schema = NodeMetadata.model_json_schema()

        assert "properties" in schema
        assert "repo" in schema["properties"]
        assert "description" in schema["properties"]["repo"]
        assert "repository" in schema["properties"]["repo"]["description"].lower()

    @pytest.mark.parametrize(
        "language",
        ["python", "javascript", "typescript", "rust", "go", "java", "cpp", "c", "ruby", "yaml", "json"],
    )
    def test_should_accept_various_languages(self, valid_node_metadata_data, language):
        """Test NodeMetadata accepts various programming languages."""
        data = valid_node_metadata_data.copy()
        data["language"] = language

        metadata = NodeMetadata(**data)

        assert metadata.language == language

    @pytest.mark.parametrize(
        "node_type",
        ["function", "class", "method", "module", "interface", "struct", "enum", "variable", "constant"],
    )
    def test_should_accept_various_node_types(self, valid_node_metadata_data, node_type):
        """Test NodeMetadata accepts various node types."""
        data = valid_node_metadata_data.copy()
        data["node_type"] = node_type

        metadata = NodeMetadata(**data)

        assert metadata.node_type == node_type

    def test_should_accept_extra_attributes(self, valid_node_metadata_data):
        """Test NodeMetadata accepts extra attributes dict."""
        data = valid_node_metadata_data.copy()
        data["extra"] = {"decorator": "@staticmethod", "visibility": "public"}

        metadata = NodeMetadata(**data)

        assert metadata.extra == {"decorator": "@staticmethod", "visibility": "public"}


class TestNode:
    """Tests for Node model."""

    def test_should_create_node_with_valid_data(self, valid_node_data):
        """Test Node creation with all valid fields."""
        node = Node(**valid_node_data)

        assert node.content == valid_node_data["content"]
        assert node.metadata == valid_node_data["metadata"]
        assert node.id is not None

    def test_should_generate_uuid_by_default(self, valid_node_data):
        """Test Node generates UUID v4 by default for id field."""
        node = Node(**valid_node_data)

        assert isinstance(node.id, uuid.UUID)
        assert node.id.version == 4

    def test_should_accept_custom_uuid(self, valid_node_data):
        """Test Node accepts a custom UUID."""
        custom_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

        node = Node(id=custom_id, content=valid_node_data["content"], metadata=valid_node_data["metadata"])

        assert node.id == custom_id

    def test_should_serialize_to_json(self, sample_node):
        """Test Node can be serialized to JSON."""
        json_str = sample_node.model_dump_json()

        assert isinstance(json_str, str)
        assert "my_function" in json_str
        assert str(sample_node.id) in json_str

    def test_should_create_placeholder_node(self):
        """Test Node.from_parsed creates a placeholder node for empty content."""
        metadata = NodeMetadata(
            repo="my-repo",
            repo_path="/path/to/repo",
            document_path="src/file.py",
            document_hash="abc123",
            language="python",
            node_type="function",
            start_byte=0,
            end_byte=0,
            start_line=1,
            end_line=1,
        )
        node = Node.from_parsed(content="", metadata=metadata)

        assert node.content == ""
        assert node.metadata.repo == "my-repo"
        assert node.metadata.repo_path == "/path/to/repo"
        assert node.metadata.document_path == "src/file.py"
        assert node.metadata.node_type == "__PLACEHOLDER__"
        assert node.metadata.language == "python"
        assert node.metadata.start_byte == 0
        assert node.metadata.end_byte == 0
        assert node.metadata.start_line == 1
        assert node.metadata.end_line == 1

    def test_should_create_placeholder_with_unique_id(self):
        """Test Node.from_parsed creates nodes with unique IDs."""
        metadata1 = NodeMetadata(
            repo="r",
            repo_path="/p",
            document_path="f.py",
            document_hash="hash1",
            language="python",
            node_type="function",
            start_byte=0,
            end_byte=0,
            start_line=1,
            end_line=1,
        )
        metadata2 = NodeMetadata(
            repo="r",
            repo_path="/p",
            document_path="f.py",
            document_hash="hash2",
            language="python",
            node_type="function",
            start_byte=0,
            end_byte=0,
            start_line=1,
            end_line=1,
        )
        node1 = Node.from_parsed(content="", metadata=metadata1)
        node2 = Node.from_parsed(content="", metadata=metadata2)

        assert node1.id != node2.id

    def test_should_create_regular_node_with_nonempty_content(self):
        """Test Node.from_parsed creates a regular node when content is provided."""
        metadata = NodeMetadata(
            repo="my-repo",
            repo_path="/path/to/repo",
            document_path="src/file.py",
            document_hash="abc123",
            language="python",
            node_type="function",
            node_name="my_function",
            start_byte=10,
            end_byte=100,
            start_line=5,
            end_line=15,
            documentation="A test function",
        )

        node = Node.from_parsed(content="def my_function():\n    pass", metadata=metadata)

        # Should use the provided metadata, not create a placeholder
        assert node.content == "def my_function():\n    pass"
        assert node.metadata.node_type == "function"  # Not __PLACEHOLDER__
        assert node.metadata.language == "python"
        assert node.metadata.node_name == "my_function"
        assert node.metadata.start_byte == 10
        assert node.metadata.end_byte == 100
        assert node.metadata.start_line == 5
        assert node.metadata.end_line == 15
        assert node.metadata.documentation == "A test function"

    @pytest.mark.parametrize(
        "content,expected_type,original_type",
        [
            ("", "__PLACEHOLDER__", "function"),  # Empty string mutates to placeholder
            ("def foo():\n    pass", "function", "function"),  # Non-empty keeps type
            ("class MyClass:\n    pass", "class", "class"),  # Non-empty keeps type
            ("x = 42", "variable", "variable"),  # Non-empty keeps type
        ],
    )
    def test_should_determine_node_type_based_on_content(self, content, expected_type, original_type):
        """Test Node.from_parsed creates different node types based on content value."""
        metadata = NodeMetadata(
            repo="test-repo",
            repo_path="/test",
            document_path="test.py",
            document_hash="abc123",
            language="python",
            node_type=original_type,
            start_byte=0,
            end_byte=len(content),
            start_line=1,
            end_line=1,
        )

        node = Node.from_parsed(content=content, metadata=metadata)

        assert node.content == content
        assert node.metadata.node_type == expected_type

    def test_should_convert_to_payload(self, sample_node):
        """Test Node.as_payload returns correct dictionary structure."""
        payload = sample_node.as_payload()

        assert payload["content"] == sample_node.content
        assert payload["document_hash"] == sample_node.metadata.document_hash
        assert payload["repo"] == sample_node.metadata.repo
        assert payload["document_path"] == sample_node.metadata.document_path
        assert payload["language"] == sample_node.metadata.language
        assert payload["node_type"] == sample_node.metadata.node_type

    def test_payload_should_not_include_id(self, sample_node):
        """Test Node.as_payload does not include the id field."""
        payload = sample_node.as_payload()

        assert "id" not in payload

    def test_payload_should_include_all_metadata_fields(self, sample_node):
        """Test Node.as_payload includes all metadata fields."""
        payload = sample_node.as_payload()

        assert "repo" in payload
        assert "repo_path" in payload
        assert "document_path" in payload
        assert "document_hash" in payload
        assert "language" in payload
        assert "node_type" in payload
        assert "node_name" in payload
        assert "start_byte" in payload
        assert "end_byte" in payload
        assert "start_line" in payload
        assert "end_line" in payload
        assert "documentation" in payload
        assert "parent_scope" in payload
        assert "signature" in payload
        assert "extra" in payload


class TestNodeIntegration:
    """Integration tests for Node model."""

    def test_should_roundtrip_through_dict_serialization(self, sample_node):
        """Test Node can be serialized and deserialized through dict."""
        data = sample_node.model_dump()
        restored = Node.model_validate(data)

        assert restored.content == sample_node.content
        assert restored.metadata.repo == sample_node.metadata.repo
        assert restored.metadata.document_hash == sample_node.metadata.document_hash

    def test_should_create_multiple_nodes_with_same_metadata_structure(self, sample_node_metadata):
        """Test multiple nodes can share the same metadata structure pattern."""
        nodes = [Node(content=f"def func{i}(): pass", metadata=sample_node_metadata) for i in range(3)]

        assert len(nodes) == 3
        assert all(node.metadata.repo == "test-repo" for node in nodes)
        assert len(set(node.id for node in nodes)) == 3  # All unique IDs

    def test_should_handle_large_content(self, sample_node_metadata):
        """Test Node handles large content correctly."""
        large_content = "x" * 100000

        node = Node(content=large_content, metadata=sample_node_metadata)

        assert len(node.content) == 100000

    def test_should_handle_unicode_content(self, sample_node_metadata):
        """Test Node handles unicode content correctly."""
        unicode_content = "def greet(): return 'こんにちは世界'"

        node = Node(content=unicode_content, metadata=sample_node_metadata)

        assert node.content == unicode_content
