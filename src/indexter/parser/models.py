import uuid

from pydantic import BaseModel, Field


class NodeMetadata(BaseModel):
    """
    Metadata describing a parsed code node's location and context.

    Contains all contextual information about a code node including its location
    within the source file, the repository it belongs to, and language-specific
    attributes like documentation and signatures.
    """

    repo: str = Field(description="Name of the repository containing the node")
    repo_path: str = Field(description="Absolute path to the repository root")
    document_path: str = Field(description="Relative path to the source file within the repository")
    hash: str | None = Field(default=None, description="Hash of the parent document")
    language: str = Field(description="Programming language of the node")
    node_type: str = Field(description="Type of code construct (function, class, etc.)")
    node_name: str | None = Field(default=None, description="Name identifier of the node")
    start_byte: int = Field(description="Starting byte offset of the node in the document")
    end_byte: int = Field(description="Ending byte offset of the node in the document")
    start_line: int = Field(description="Starting line number (1-indexed) in the document")
    end_line: int = Field(description="Ending line number (1-indexed) in the document")
    documentation: str | None = Field(default=None, description="Docstring, comments, or other documentation text")
    parent_scope: str | None = Field(
        default=None, description="Enclosing scope or class name (e.g., 'MyClass' for methods)"
    )
    signature: str | None = Field(
        default=None, description="Function/method signature with parameters and return types"
    )
    extra: dict[str, str] = Field(
        default_factory=dict, description="Language-specific attributes (e.g., decorators, modifiers, attributes)"
    )


class Node(BaseModel):
    """
    A parsed code node with content and metadata, ready for embedding.

    Represents a semantic unit of code (function, class, etc.) that has been
    extracted from source files and prepared for vector embedding and storage.
    Each node has a unique identifier, the actual code content, and rich metadata.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the node (UUID v4)")
    content: str = Field(description="Source content of the node")
    metadata: NodeMetadata = Field(description="Metadata describing the node's context and location")

    @classmethod
    def placeholder(cls, repo: str, repo_path: str, document_path: str, hash: str | None = None) -> "Node":
        """Create a placeholder node with default values."""
        return cls(
            content="",
            metadata=NodeMetadata(
                repo=repo,
                repo_path=repo_path,
                document_path=document_path,
                hash=hash,
                language="",
                node_type="__PLACEHOLDER__",
                node_name="",
                start_byte=0,
                end_byte=0,
                start_line=0,
                end_line=0,
            ),
        )

    def as_payload(self) -> dict:
        """Convert the node to a payload dictionary for storage in the vector store."""
        return {
            "content": self.content,
            **self.metadata.model_dump(),
        }
