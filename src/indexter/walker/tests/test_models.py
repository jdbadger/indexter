import json

import pytest
from pydantic import ValidationError

from indexter.walker.models import Document, DocumentMetadata


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    @pytest.fixture
    def valid_metadata_data(self):
        """Fixture providing valid metadata data."""
        return {
            "repo": "test-repo",
            "repo_path": "/home/user/repos/test-repo",
            "hash": "abc123def456",
            "ext": ".py",
            "size_bytes": 1024,
            "mtime": 1234567890.5,
        }

    def test_should_create_metadata_with_valid_data(self, valid_metadata_data):
        """Test DocumentMetadata creation with all valid fields."""
        metadata = DocumentMetadata(**valid_metadata_data)

        assert metadata.repo == "test-repo"
        assert metadata.repo_path == "/home/user/repos/test-repo"
        assert metadata.hash == "abc123def456"
        assert metadata.ext == ".py"
        assert metadata.size_bytes == 1024
        assert metadata.mtime == 1234567890.5

    @pytest.mark.parametrize(
        "field_name",
        ["repo", "repo_path", "hash", "ext", "size_bytes", "mtime"],
    )
    def test_should_reject_missing_required_field(self, valid_metadata_data, field_name):
        """Test DocumentMetadata rejects missing required fields."""
        data = valid_metadata_data.copy()
        del data[field_name]

        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(**data)

        assert field_name in str(exc_info.value)

    @pytest.mark.parametrize(
        "ext",
        [".py", ".js", ".ts", ".rs", ".go", ".java", ".cpp", ".c", ".h", ".md", ".txt", ""],
    )
    def test_should_accept_various_file_extensions(self, valid_metadata_data, ext):
        """Test DocumentMetadata accepts various file extensions."""
        data = valid_metadata_data.copy()
        data["ext"] = ext

        metadata = DocumentMetadata(**data)

        assert metadata.ext == ext

    @pytest.mark.parametrize(
        "size_bytes",
        [0, 1, 100, 1024, 1048576, 10485760],
    )
    def test_should_accept_various_file_sizes(self, valid_metadata_data, size_bytes):
        """Test DocumentMetadata accepts various file sizes."""
        data = valid_metadata_data.copy()
        data["size_bytes"] = size_bytes

        metadata = DocumentMetadata(**data)

        assert metadata.size_bytes == size_bytes

    def test_should_reject_invalid_size_type(self, valid_metadata_data):
        """Test DocumentMetadata rejects invalid size_bytes type."""
        data = valid_metadata_data.copy()
        data["size_bytes"] = "not_a_number"

        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(**data)

        assert "size_bytes" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "mtime",
        [0.0, 1234567890.0, 1234567890.123456, 1700000000.0, 2000000000.0],
    )
    def test_should_accept_various_modification_times(self, valid_metadata_data, mtime):
        """Test DocumentMetadata accepts various modification times."""
        data = valid_metadata_data.copy()
        data["mtime"] = mtime

        metadata = DocumentMetadata(**data)

        assert metadata.mtime == mtime

    def test_should_reject_invalid_mtime_type(self, valid_metadata_data):
        """Test DocumentMetadata rejects invalid mtime type."""
        data = valid_metadata_data.copy()
        data["mtime"] = "not_a_timestamp"

        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(**data)

        assert "mtime" in str(exc_info.value).lower()

    def test_should_accept_absolute_and_relative_repo_paths(self, valid_metadata_data):
        """Test DocumentMetadata accepts different repo path formats."""
        absolute_path = "/home/user/repos/test-repo"
        data = valid_metadata_data.copy()
        data["repo_path"] = absolute_path

        metadata = DocumentMetadata(**data)

        assert metadata.repo_path == absolute_path

    def test_should_serialize_to_dict(self, valid_metadata_data):
        """Test DocumentMetadata can be serialized to dict."""
        metadata = DocumentMetadata(**valid_metadata_data)

        result = metadata.model_dump()

        assert result == valid_metadata_data
        assert isinstance(result, dict)

    def test_should_serialize_to_json(self, valid_metadata_data):
        """Test DocumentMetadata can be serialized to JSON."""
        metadata = DocumentMetadata(**valid_metadata_data)

        json_str = metadata.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-repo" in json_str
        assert "abc123def456" in json_str

    def test_should_deserialize_from_dict(self, valid_metadata_data):
        """Test DocumentMetadata can be deserialized from dict."""
        metadata = DocumentMetadata.model_validate(valid_metadata_data)

        assert metadata.repo == valid_metadata_data["repo"]
        assert metadata.hash == valid_metadata_data["hash"]

    def test_should_support_field_descriptions(self):
        """Test DocumentMetadata fields have descriptions."""
        schema = DocumentMetadata.model_json_schema()

        assert "properties" in schema
        assert "repo" in schema["properties"]
        assert "description" in schema["properties"]["repo"]
        assert "repository" in schema["properties"]["repo"]["description"].lower()


class TestDocument:
    """Tests for Document model."""

    @pytest.fixture
    def valid_metadata(self):
        """Fixture providing valid DocumentMetadata."""
        return DocumentMetadata(
            repo="test-repo",
            repo_path="/home/user/repos/test-repo",
            hash="abc123def456",
            ext=".py",
            size_bytes=1024,
            mtime=1234567890.5,
        )

    @pytest.fixture
    def valid_document_data(self, valid_metadata):
        """Fixture providing valid document data."""
        return {
            "path": "src/main.py",
            "content": "print('Hello, World!')",
            "metadata": valid_metadata,
        }

    def test_should_create_document_with_valid_data(self, valid_document_data):
        """Test Document creation with all valid fields."""
        doc = Document(**valid_document_data)

        assert doc.path == "src/main.py"
        assert doc.content == "print('Hello, World!')"
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.metadata.repo == "test-repo"

    @pytest.mark.parametrize(
        "field_name",
        ["path", "content", "metadata"],
    )
    def test_should_reject_missing_required_field(self, valid_document_data, field_name):
        """Test Document rejects missing required fields."""
        data = valid_document_data.copy()
        del data[field_name]

        with pytest.raises(ValidationError) as exc_info:
            Document(**data)

        assert field_name in str(exc_info.value)

    @pytest.mark.parametrize(
        "path",
        [
            "main.py",
            "src/main.py",
            "src/utils/helpers.py",
            "packages/core/src/index.ts",
            "README.md",
            "a/b/c/d/e/f/g.txt",
        ],
    )
    def test_should_accept_various_file_paths(self, valid_document_data, path):
        """Test Document accepts various file paths."""
        data = valid_document_data.copy()
        data["path"] = path

        doc = Document(**data)

        assert doc.path == path

    def test_should_accept_empty_content(self, valid_document_data):
        """Test Document accepts empty content."""
        data = valid_document_data.copy()
        data["content"] = ""

        doc = Document(**data)

        assert doc.content == ""

    def test_should_accept_multiline_content(self, valid_document_data):
        """Test Document accepts multiline content."""
        content = """def hello():
    print('Hello, World!')
    return True

if __name__ == '__main__':
    hello()
"""
        data = valid_document_data.copy()
        data["content"] = content

        doc = Document(**data)

        assert doc.content == content
        assert "\n" in doc.content

    def test_should_accept_unicode_content(self, valid_document_data):
        """Test Document accepts unicode content."""
        content = "# Comment with unicode: café, naïve, 日本語, 🚀"
        data = valid_document_data.copy()
        data["content"] = content

        doc = Document(**data)

        assert doc.content == content
        assert "café" in doc.content
        assert "🚀" in doc.content

    def test_should_accept_large_content(self, valid_document_data):
        """Test Document accepts large content."""
        # Create a large content string (10KB)
        content = "x" * 10240
        data = valid_document_data.copy()
        data["content"] = content

        doc = Document(**data)

        assert doc.content == content
        assert len(doc.content) == 10240

    def test_should_accept_metadata_as_dict(self, valid_document_data):
        """Test Document accepts metadata as dict."""
        data = valid_document_data.copy()
        data["metadata"] = {
            "repo": "test-repo",
            "repo_path": "/home/user/repos/test-repo",
            "hash": "def789",
            "ext": ".js",
            "size_bytes": 2048,
            "mtime": 1234567890.0,
        }

        doc = Document(**data)

        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.metadata.repo == "test-repo"
        assert doc.metadata.hash == "def789"

    def test_should_reject_invalid_metadata(self, valid_document_data):
        """Test Document rejects invalid metadata."""
        data = valid_document_data.copy()
        data["metadata"] = {"invalid": "data"}

        with pytest.raises(ValidationError) as exc_info:
            Document(**data)

        error_str = str(exc_info.value).lower()
        assert "metadata" in error_str

    def test_should_serialize_to_dict(self, valid_document_data):
        """Test Document can be serialized to dict."""
        doc = Document(**valid_document_data)

        result = doc.model_dump()

        assert isinstance(result, dict)
        assert result["path"] == "src/main.py"
        assert result["content"] == "print('Hello, World!')"
        assert isinstance(result["metadata"], dict)
        assert result["metadata"]["repo"] == "test-repo"

    def test_should_serialize_to_json(self, valid_document_data):
        """Test Document can be serialized to JSON."""
        doc = Document(**valid_document_data)

        json_str = doc.model_dump_json()

        assert isinstance(json_str, str)
        assert "src/main.py" in json_str
        assert "Hello, World!" in json_str
        assert "test-repo" in json_str

    def test_should_deserialize_from_dict(self, valid_document_data):
        """Test Document can be deserialized from dict."""
        data = valid_document_data.copy()
        data["metadata"] = data["metadata"].model_dump()

        doc = Document.model_validate(data)

        assert doc.path == "src/main.py"
        assert doc.content == "print('Hello, World!')"
        assert doc.metadata.repo == "test-repo"

    def test_should_support_field_descriptions(self):
        """Test Document fields have descriptions."""
        schema = Document.model_json_schema()

        assert "properties" in schema
        assert "path" in schema["properties"]
        assert "description" in schema["properties"]["path"]
        assert "relative" in schema["properties"]["path"]["description"].lower()

    def test_should_handle_special_characters_in_path(self, valid_document_data):
        """Test Document handles special characters in path."""
        paths = [
            "file with spaces.py",
            "file-with-dashes.py",
            "file_with_underscores.py",
            "file.test.py",
        ]

        for path in paths:
            data = valid_document_data.copy()
            data["path"] = path

            doc = Document(**data)

            assert doc.path == path

    def test_should_preserve_content_whitespace(self, valid_document_data):
        """Test Document preserves whitespace in content."""
        content = "  \t  indented content  \t  \n  \n"
        data = valid_document_data.copy()
        data["content"] = content

        doc = Document(**data)

        assert doc.content == content


class TestDocumentMetadataIntegration:
    """Integration tests for DocumentMetadata."""

    def test_should_roundtrip_through_dict_serialization(self):
        """Test metadata can roundtrip through dict serialization."""
        original = DocumentMetadata(
            repo="integration-repo",
            repo_path="/path/to/repo",
            hash="hash123",
            ext=".ts",
            size_bytes=4096,
            mtime=1234567890.123,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = DocumentMetadata.model_validate(data)

        assert restored.repo == original.repo
        assert restored.repo_path == original.repo_path
        assert restored.hash == original.hash
        assert restored.ext == original.ext
        assert restored.size_bytes == original.size_bytes
        assert restored.mtime == original.mtime

    def test_should_roundtrip_through_json_serialization(self):
        """Test metadata can roundtrip through JSON serialization."""
        original = DocumentMetadata(
            repo="json-repo",
            repo_path="/json/path",
            hash="jsonhash",
            ext=".json",
            size_bytes=512,
            mtime=1700000000.0,
        )

        # Serialize to JSON and deserialize
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        restored = DocumentMetadata.model_validate(data)

        assert restored.repo == original.repo
        assert restored.hash == original.hash


class TestDocumentIntegration:
    """Integration tests for Document."""

    def test_should_roundtrip_through_dict_serialization(self):
        """Test document can roundtrip through dict serialization."""
        original = Document(
            path="integration/test.py",
            content="# Integration test\npass",
            metadata=DocumentMetadata(
                repo="test-repo",
                repo_path="/test/path",
                hash="inthash",
                ext=".py",
                size_bytes=100,
                mtime=1234567890.0,
            ),
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = Document.model_validate(data)

        assert restored.path == original.path
        assert restored.content == original.content
        assert restored.metadata.repo == original.metadata.repo
        assert restored.metadata.hash == original.metadata.hash

    def test_should_roundtrip_through_json_serialization(self):
        """Test document can roundtrip through JSON serialization."""
        original = Document(
            path="json/test.js",
            content='console.log("test");',
            metadata=DocumentMetadata(
                repo="js-repo",
                repo_path="/js/path",
                hash="jshash",
                ext=".js",
                size_bytes=200,
                mtime=1700000000.0,
            ),
        )

        # Serialize to JSON and deserialize
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        restored = Document.model_validate(data)

        assert restored.path == original.path
        assert restored.content == original.content
        assert restored.metadata.repo == original.metadata.repo

    def test_should_create_multiple_documents_with_same_metadata_structure(self):
        """Test creating multiple documents with consistent metadata structure."""
        metadata_template = {
            "repo": "multi-doc-repo",
            "repo_path": "/multi/doc",
            "ext": ".py",
            "size_bytes": 500,
            "mtime": 1234567890.0,
        }

        documents = []
        for i in range(5):
            metadata = DocumentMetadata(**{**metadata_template, "hash": f"hash{i}"})
            doc = Document(
                path=f"file{i}.py",
                content=f"# File {i}",
                metadata=metadata,
            )
            documents.append(doc)

        assert len(documents) == 5
        for i, doc in enumerate(documents):
            assert doc.path == f"file{i}.py"
            assert doc.metadata.hash == f"hash{i}"
            assert doc.metadata.repo == "multi-doc-repo"

    def test_should_handle_nested_serialization(self):
        """Test nested serialization of Document with embedded metadata."""
        doc = Document(
            path="nested/test.py",
            content="nested content",
            metadata=DocumentMetadata(
                repo="nested-repo",
                repo_path="/nested",
                hash="nestedhash",
                ext=".py",
                size_bytes=128,
                mtime=1234567890.0,
            ),
        )

        # Full serialization
        data = doc.model_dump()

        # Verify structure
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)
        assert "repo" in data["metadata"]
        assert data["metadata"]["repo"] == "nested-repo"

    def test_should_validate_complete_document_structure(self):
        """Test validation of complete document with all fields."""
        doc = Document(
            path="complete/document.rs",
            content='fn main() { println!("Hello"); }',
            metadata=DocumentMetadata(
                repo="complete-repo",
                repo_path="/complete/path/to/repo",
                hash="completehash123456789",
                ext=".rs",
                size_bytes=1024,
                mtime=1700000000.5,
            ),
        )

        # Verify all fields are accessible
        assert doc.path == "complete/document.rs"
        assert "fn main()" in doc.content
        assert doc.metadata.repo == "complete-repo"
        assert doc.metadata.repo_path == "/complete/path/to/repo"
        assert doc.metadata.hash == "completehash123456789"
        assert doc.metadata.ext == ".rs"
        assert doc.metadata.size_bytes == 1024
        assert doc.metadata.mtime == 1700000000.5
