from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata for a source document within a repository.

    Contains information about a source file including its path, size,
    modification time, and content hash for change detection.
    """

    repo: str = Field(description="Name of the repository containing the document")
    repo_path: str = Field(description="Absolute path to the repository")
    hash: str = Field(description="SHA-256 hash of the relative path + file content")
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
    metadata: DocumentMetadata = Field(
        description="Metadata about the document including size, modification time, and hash"
    )
