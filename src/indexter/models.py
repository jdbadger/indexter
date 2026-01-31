"""Domain models for Indexter's indexing and search pipeline.

This module defines the Pydantic models that flow through Indexter's core
pipeline: file discovery, parsing, indexing, and search. All models are
immutable data containers with no I/O or side effects.

Pipeline Flow:
    Walker yields documents → Parser extracts nodes → Store indexes nodes

    Document / DocumentMetadata:
        A source file discovered by the Walker, with its content and
        file-system metadata. A computed SHA-256 hash (path + content)
        enables incremental change detection.

    Node / NodeMetadata:
        A semantic code unit (function, class, heading, etc.) extracted by
        a language-specific Parser. Each node carries rich metadata including
        location, documentation, signature, and parent scope.

    IndexResult:
        Statistics returned after an indexing operation: documents
        indexed/deleted, nodes added/deleted, duration, and errors.

    SearchResult / SearchResults:
        Results from hybrid semantic + keyword search over indexed nodes.
        Each result contains the matched source content, a relevance score,
        and the node's metadata.

    RepoMetadata:
        Aggregate statistics about an indexed repository: document count,
        node count, languages, node types, and an ASCII file tree.

Example:
    Creating a document and parsing it into nodes::

        from indexter.models import Document, DocumentMetadata, Node

        doc = Document(
            path="src/utils.py",
            content="def greet(name): ...",
            metadata=DocumentMetadata(
                repo="my-repo",
                repo_path="/home/user/my-repo",
                ext=".py",
                size_bytes=42,
                mtime=1700000000.0,
            ),
        )
        print(doc.hash)  # SHA-256 of "src/utils.py:def greet(name): ..."
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)


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
    document_hash: str = Field(description="Hash of the source document for change detection")
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
    def from_parsed(cls, content: str, metadata: NodeMetadata) -> Node:
        """
        Create a Node instance from parsed content and metadata.
        If the content is empty, return a placeholder node."""
        if not content:
            metadata.node_type = "__PLACEHOLDER__"
        return cls(content=content, metadata=metadata)

    def as_payload(self) -> dict:
        """Convert the node to a payload dictionary for storage in the vector store."""
        return {
            "content": self.content,
            **self.metadata.model_dump(),
        }


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


class IndexResult(BaseModel):
    """
    Result of a repository indexing/sync operation.

    Tracks statistics and outcomes from parsing and indexing a repository,
    including file counts, node counts, errors, and timing information.
    """

    repo: str = Field(description="Name of the repository indexed")
    repo_path: str = Field(description="Path to the repository indexed")
    documents_indexed: list[str] = Field(
        default_factory=list,
        description="List of file paths that were successfully indexed",
    )
    documents_deleted: list[str] = Field(
        default_factory=list,
        description="List of file paths that were deleted from the index",
    )
    nodes_added: int = Field(default=0, description="Count of new code nodes added to the index")
    nodes_deleted: int = Field(default=0, description="Count of code nodes deleted from the index")
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration: float = Field(default=0.0, description="Duration of the indexing operation in seconds")
    errors: list[str] = Field(default_factory=list, description="List of error messages encountered during indexing")

    @computed_field
    @property
    def summary(self) -> str:
        """Summary of the indexing result."""

        return (
            f"Indexed {len(self.documents_indexed)} documents (+{self.nodes_added} nodes added, "
            f"-{self.nodes_deleted} nodes deleted) "
            f"in {self.duration:.2f}s"
        )


class SearchResult(BaseModel):
    """
    A single search result from semantic code search.

    Represents one matching code chunk with its similarity score and metadata.
    """

    content: str = Field(description="Source code content")
    score: float = Field(description="Similarity score (0.0-1.0)")
    metadata: dict[str, Any] = Field(description="Metadata about the search result")


class SearchResults(BaseModel):
    """
    Response from repository semantic search.

    Contains the list of matched nodes along with query metadata.
    """

    repo: str | None = Field(default=None, description="Name of the repository searched")
    repo_path: str | None = Field(default=None, description="Path to the repository searched")
    results: list[SearchResult] = Field(description="Matched nodes")
    query: str = Field(description="Original search query")
    filters: dict[str, Any] = Field(description="Applied search filters")

    @computed_field
    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.results)


class RepoMetadata(BaseModel):
    """
    Status information for an indexed repository.

    Provides current state of repository indexing including node counts,
    document counts, and staleness indicators.
    """

    document_paths: list[str] = Field(
        default_factory=list, description="List of indexed document paths (relative to Repo root)"
    )
    documents: int = Field(default=0, description="Number of indexed documents")
    node_types: list[str] = Field(default_factory=list, description="Indexed node types")
    nodes: int = Field(default=0, description="Number of nodes")
    languages: list[str] = Field(default_factory=list, description="Indexed languages")

    @computed_field
    @property
    def document_tree(self) -> str:
        """Hierarchical ASCII tree representation of indexed documents."""
        if not self.document_paths:
            return "(no documents)"

        # Build nested tree structure from paths
        tree: dict[str, dict] = {}
        for path in sorted(self.document_paths):
            parts = path.split("/")
            current = tree
            for part in parts:
                current = current.setdefault(part, {})

        # Render tree with ASCII box-drawing characters
        lines: list[str] = []

        def render(node: dict[str, dict], prefix: str = "") -> None:
            """Recursively render tree nodes with proper connectors."""
            items = list(node.items())
            for i, (name, children) in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                # Add trailing / for directories (nodes with children)
                display_name = f"{name}/" if children else name
                lines.append(f"{prefix}{connector}{display_name}")
                if children:
                    extension = "    " if is_last else "│   "
                    render(children, prefix + extension)

        render(tree)
        return "\n".join(lines)
