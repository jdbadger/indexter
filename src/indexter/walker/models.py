import hashlib

from pydantic import BaseModel, Field, computed_field


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of the provided content."""
    return hashlib.sha256(content.encode()).hexdigest()


class DocumentMetadata(BaseModel):
    """
    Metadata for a source document within a repository.

    Contains information about a source file including its repository,
    file extension, size, and modification time.
    """

    repo: str = Field(description="Name of the repository containing the document")
    repo_path: str = Field(description="Absolute path to the repository")
    ext: str = Field(description="File extension (e.g., .py, .js)")
    size_bytes: int = Field(description="File size in bytes")
    mtime: float = Field(description="Modification time as Unix timestamp")


class Document(BaseModel):
    """
    A source code file with metadata for change detection.

    Represents a file from the repository with its content and metadata,
    including a content hash for efficient change detection during indexing.
    """

    path: str = Field(description="Relative path to the document within the repository")
    content: str = Field(description="Full text content of the document")
    metadata: DocumentMetadata = Field(description="Metadata about the document.")

    @computed_field
    @property
    def hash(self) -> str:
        """Compute a hash for the document based on its path and content."""
        hash_input = f"{self.path}:{self.content}"
        return compute_hash(hash_input)
