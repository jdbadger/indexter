from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


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
    documents_checked: int = Field(default=0, description="Total number of documents checked")
    skipped_documents: int = Field(default=0, description="Number of documents skipped due to max_files limit")
    nodes_added: int = Field(default=0, description="Count of new code nodes added to the index")
    nodes_updated: int = Field(default=0, description="Count of code nodes updated (re-indexed)")
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration: float = Field(default=0.0, description="Duration of the indexing operation in seconds")
    errors: list[str] = Field(default_factory=list, description="List of error messages encountered during indexing")

    @computed_field
    @property
    def summary(self) -> str:
        """Summary of the indexing result."""

        return (
            f"Indexed {len(self.documents_indexed)} documents (+{self.nodes_added} nodes added, "
            f"~{self.nodes_updated} nodes updated) "
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
