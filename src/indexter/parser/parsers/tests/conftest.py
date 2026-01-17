import pytest

from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def sample_document_metadata():
    """Create a sample DocumentMetadata instance for testing."""
    return DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        hash="abc123def456",
        ext=".py",
        size_bytes=1024,
        mtime=1234567890.0,
    )


@pytest.fixture
def sample_document(sample_document_metadata):
    """Create a sample Document instance for testing."""
    return Document(
        path="test/file.py",
        content="# Sample Python file\n\ndef hello():\n    print('hello')\n",
        metadata=sample_document_metadata,
    )
