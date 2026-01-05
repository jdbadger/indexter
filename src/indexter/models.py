"""
Core data models and repository management for Indexter.

This module provides the primary API for interacting with Indexter through both
the CLI and MCP (Model Context Protocol) server. It defines the data models for
code indexing and implements the repository management system.

Architecture
------------
The module is built around a few key concepts:

1. **Repository Management**: The `Repo` class manages Git repositories,
   handling registration, configuration, and lifecycle operations.

2. **Incremental Indexing**: Uses content hashing for efficient change detection,
   only re-parsing files that have been modified since the last index operation.

3. **Code Parsing**: Extracts semantic units (functions, classes, methods) from
   source files and stores them as `Node` objects with rich metadata.

4. **Vector Storage**: Nodes are embedded and stored in a vector database for
   semantic code search across repositories.

Main Classes
------------
Repo:
    The primary interface for repository operations. Provides class methods for
    managing repositories (init, get, list, remove) and instance methods for
    indexing and searching (index, search, status, get_document_hashes).

IndexResult:
    Tracks statistics from an indexing operation, including file and node counts,
    errors, and timing information.

Node:
    Represents a parsed code construct (function, class, etc.) with its source
    code content and comprehensive metadata ready for embedding.

NodeMetadata:
    Contains contextual information about a Node including location, language,
    type, documentation, and language-specific attributes.

Document:
    Represents a source file with content and metadata, including a SHA-256 hash
    for change detection during incremental indexing.

Workflow
--------
Typical usage follows this pattern:

1. Initialize a repository:
   >>> repo = await Repo.init(Path("/path/to/project"))

2. Index the repository (incremental by default):
   >>> result = await repo.index()
   >>> print(f"Indexed {result.nodes_added} nodes from {len(result.files_indexed)} files")

3. Search the indexed code:
   >>> results = await repo.search("authentication logic", limit=5)
   >>> for hit in results:
   ...     print(hit['metadata']['node_name'], hit['score'])

4. Check repository status:
   >>> status = await repo.status()
   >>> print(f"{status['documents_indexed_stale']} files need re-indexing")

5. Full re-index when needed:
   >>> result = await repo.index(full=True)

Change Detection
----------------
The indexing system uses SHA-256 content hashing to efficiently detect changes:

- **New files**: Parse and add nodes to the vector store
- **Modified files**: Delete old nodes, re-parse, and add new nodes
- **Deleted files**: Remove associated nodes from the vector store
- **Unchanged files**: Skip processing entirely

This approach minimizes redundant parsing and keeps the index synchronized with
the repository state while maintaining fast incremental updates.

Configuration
-------------
Repository behavior is controlled through `RepoSettings`:

- Ignore patterns (similar to .gitignore)
- Maximum file size limits
- Maximum number of files to process per index operation
- Batch sizes for vector store operations
- Custom embedding models

Settings can be specified globally or per-repository via `indexter.toml` or
`pyproject.toml` files.

Examples
--------
Initialize multiple repositories:
    >>> repo1 = await Repo.init(Path("~/projects/backend"))
    >>> repo2 = await Repo.init(Path("~/projects/frontend"))
    >>> repos = await Repo.list()
    >>> print(f"Managing {len(repos)} repositories")

Search across specific file types:
    >>> results = await repo.search(
    ...     query="user authentication",
    ...     language="python",
    ...     node_type="function",
    ...     has_documentation=True
    ... )

Monitor indexing progress:
    >>> status = await repo.status()
    >>> if status['documents_indexed_stale'] > 0:
    ...     result = await repo.index()
    ...     print(f"Updated {result.nodes_updated} nodes")

Notes
-----
- All repository operations are asynchronous and require an event loop
- The vector store is managed transparently; see `store.py` for details
- File walking respects .gitignore and custom ignore patterns
- Binary, minified, and oversized files are automatically skipped
"""

from __future__ import annotations

import builtins
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from .config import RepoSettings
from .exceptions import RepoExistsError, RepoNotFoundError
from .parsers import get_parser
from .store import store
from .walker import Walker

logger = logging.getLogger(__name__)


class IndexResult(BaseModel):
    """
    Result of a repository indexing/sync operation.

    Tracks statistics and outcomes from parsing and indexing a repository,
    including file counts, node counts, errors, and timing information.

    Attributes:
        files_indexed: List of file paths that were successfully indexed.
        files_deleted: List of file paths that were deleted from the index.
        files_checked: Total number of files examined during the sync.
        skipped_files: Number of files skipped due to max_files limit.
        nodes_added: Count of new code nodes added to the index.
        nodes_deleted: Count of code nodes removed from the index.
        nodes_updated: Count of code nodes updated (re-indexed).
        indexed_at: Timestamp when the indexing operation completed.
        errors: List of error messages encountered during indexing.
    """

    files_indexed: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    files_checked: int = 0
    skipped_files: int = 0
    nodes_added: int = 0
    nodes_deleted: int = 0
    nodes_updated: int = 0
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration: float = 0.0
    errors: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def summary(self) -> str:
        """Summary of the indexing result."""
        return (
            f"Indexed {len(self.files_indexed)} files (+{self.nodes_added} nodes, "
            f"~{self.nodes_updated} updated, -{self.nodes_deleted} deleted) "
            f"in {self.duration:.2f}s"
        )


class NodeMetadata(BaseModel):
    """
    Metadata describing a parsed code node's location and context.

    Contains all contextual information about a code node including its location
    within the source file, the repository it belongs to, and language-specific
    attributes like documentation and signatures.

    Attributes:
        hash: Content hash of the parent document for change detection.
        repo_path: Absolute path to the repository root directory.
        document_path: Relative path to the source file within the repository.
        language: Programming language of the node (e.g., 'python', 'javascript').
        node_type: Type of code construct (e.g., 'function', 'class', 'method').
        node_name: Name identifier of the node (function/class/variable name).
        start_byte: Starting byte offset of the node in the document.
        end_byte: Ending byte offset of the node in the document.
        start_line: Starting line number (1-indexed) in the document.
        end_line: Ending line number (1-indexed) in the document.
        documentation: Docstring, comments, or other documentation text.
        parent_scope: Enclosing scope or class name (e.g., 'MyClass' for methods).
        signature: Function/method signature with parameters and return types.
        extra: Language-specific attributes (e.g., decorators, modifiers, attributes).
    """

    hash: str
    repo_path: str
    document_path: str
    language: str
    node_type: str
    node_name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    documentation: str | None = None
    parent_scope: str | None = None
    signature: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class Node(BaseModel):
    """
    A parsed code node with content and metadata, ready for embedding.

    Represents a semantic unit of code (function, class, etc.) that has been
    extracted from source files and prepared for vector embedding and storage.
    Each node has a unique identifier, the actual code content, and rich metadata.

    Attributes:
        id: Unique identifier (UUID v4) for the node.
        content: The actual source code text of the node.
        metadata: Comprehensive metadata about the node's context and location.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str
    metadata: NodeMetadata


class Document(BaseModel):
    """
    A source code file with metadata for change detection.

    Represents a file from the repository with its content and metadata,
    including a content hash for efficient change detection during indexing.

    Attributes:
        path: Relative path to the file within the repository.
        size_bytes: File size in bytes.
        mtime: Modification time as Unix timestamp (seconds since epoch).
        content: Complete text content of the file.
        hash: SHA-256 hash of the file content for change detection.
    """

    path: str
    size_bytes: int
    mtime: float
    content: str
    hash: str


class Repo(BaseModel):
    """
    A code repository configured and managed by Indexter.

    Represents a Git repository that has been added to Indexter for indexing.
    Provides methods for repository management (add, remove, list) and indexing
    operations (parse, search, status).

    The repository configuration includes paths, ignore patterns, and indexing
    parameters that control how files are processed and stored.

    Attributes:
        settings: Repository-specific configuration and settings.
    """

    settings: RepoSettings

    @computed_field
    @property
    def collection_name(self) -> str:
        """Name of the VectorStore collection for this repo."""
        return self.settings.collection_name

    @computed_field
    @property
    def name(self) -> str:
        """Name of the repository."""
        return self.settings.name

    @computed_field
    @property
    def path(self) -> str:
        """Absolute path to the repository root."""
        return str(self.settings.path)

    @classmethod
    async def init(cls, path: Path) -> Repo:
        """
        Initialize and register a new repository with Indexter.

        Validates the path is a Git repository, checks for name conflicts,
        and adds the repository to the configuration. If the repository is
        already configured at the same path, returns the existing configuration.

        Args:
            path: Path to the git repository root directory.

        Returns:
            Repo instance for the initialized repository.

        Raises:
            RepoExistsError: If a different repository with the same name already
                exists. Repository names are derived from the directory name.
            ValueError: If the path is not a valid git repository (no .git directory).
        """
        repos = await RepoSettings.load()
        resolved_path = path.resolve()

        # Create new config to get the derived name
        repo_settings = RepoSettings(path=resolved_path)

        # Check if name already exists
        for existing in repos:
            if existing.name == repo_settings.name:
                if existing.path.resolve() == resolved_path:
                    # Same repo, already configured
                    logger.info(f"Repository already configured: {repo_settings.name}")
                    return cls(settings=existing)
                else:
                    # Different repo with same name
                    raise RepoExistsError(
                        f"A repository named '{existing.name}' already exists "
                        f"at {existing.path}. Rename the directory to use a unique name."
                    )

        repos.append(repo_settings)
        await RepoSettings.save(repos)

        logger.info(f"Added repository: {repo_settings.name} ({resolved_path})")
        return cls(settings=repo_settings)

    @classmethod
    async def get(cls, name: str) -> Repo:
        """
        Retrieve a configured repository by name.

        Searches the configuration for a repository matching the given name
        and returns the corresponding Repo instance.

        Args:
            name: Repository name (derived from the directory name containing .git).

        Returns:
            Repo instance for the requested repository.

        Raises:
            RepoNotFoundError: If no repository with the given name is configured.
        """
        repos = await RepoSettings.load()
        for repo_settings in repos:
            if repo_settings.name == name:
                return cls(settings=repo_settings)
        raise RepoNotFoundError(f"Repository not found: {name}")

    @classmethod
    async def list(cls) -> builtins.list[Repo]:
        """
        List all configured repositories.

        Retrieves all repositories that have been registered with Indexter.

        Returns:
            List of Repo instances for all configured repositories.
        """
        repos = await RepoSettings.load()
        return [cls(settings=repo_settings) for repo_settings in repos]

    @classmethod
    async def remove(cls, name: str) -> bool:
        """
        Remove a repository and its indexed data.

        Deletes the repository's vector store collection and removes it from
        the configuration. This operation is permanent and cannot be undone.

        Args:
            name: Name of the repository to remove.

        Returns:
            True if the repository was successfully removed, False if it was
            already removed by another process (race condition).

        Raises:
            RepoNotFoundError: If no repository with the given name exists.
        """
        repo = await cls.get(name)

        # Delete collection from store
        await store.delete_collection(repo.collection_name)

        # Remove from repos.json
        repo_settings_list = await RepoSettings.load()
        new_repo_settings_list = [r for r in repo_settings_list if r.name != name]
        await RepoSettings.save(new_repo_settings_list)
        if new_repo_settings_list != repo_settings_list:
            logger.info(f"Removed repository: {name}")
            return True
        return False

    async def get_document_hashes(self) -> dict[str, str]:
        """
        Compute content hashes for all eligible files in the repository.

        Walks the repository and computes SHA-256 hashes for all files that
        pass filtering (size limits, ignore patterns, etc.). Used for change
        detection during incremental indexing.

        Returns:
            Dictionary mapping file paths to their SHA-256 content hashes.
        """
        document_hashes: dict[str, str] = {}
        walker = Walker(self)
        async for doc in walker.walk():
            doc = Document.model_validate(doc)
            document_hashes[doc.path] = doc.hash
        return document_hashes

    async def search(
        self,
        query: str,
        file_path: str | None = None,
        language: str | None = None,
        node_type: str | None = None,
        node_name: str | None = None,
        has_documentation: bool | None = None,
        limit: int | None = None,
    ) -> builtins.list[dict]:
        """
        Perform semantic search over indexed code nodes in the repository.

        Searches the repository's vector store using embedding-based similarity.
        Results can be filtered by multiple metadata criteria to narrow down
        the search scope.

        Args:
            query: Natural language or code search query.
            limit: Maximum number of results to return. Defaults to the repository's top_k setting.
            file_path: Filter by file path. Use exact match or prefix with
                trailing '/' for directory filtering.
            language: Filter by programming language (e.g., 'python', 'javascript').
            node_type: Filter by code construct type (e.g., 'function', 'class', 'method').
            node_name: Filter by exact node name (function/class name).
            has_documentation: Filter by documentation presence. True for nodes
                with docstrings/comments, False for undocumented nodes.

        Returns:
            List of search results, each containing the node content, metadata,
            and similarity score. Ordered by relevance (highest score first).
        """
        return await store.search(
            collection_name=self.collection_name,
            query=query,
            file_path=file_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            has_documentation=has_documentation,
            limit=limit or self.settings.top_k,
        )

    async def status(self) -> dict:
        """
        Get indexing status and statistics for the repository.

        Compares the current repository state with the indexed data to identify
        stale documents (files that have been modified or deleted since last indexing).

        Returns:
            Dictionary with the following keys:
                - repository: Name of the repository.
                - path: Absolute path to the repository.
                - nodes_indexed: Total number of code nodes in the index.
                - documents_indexed: Number of documents currently indexed.
                - documents_indexed_stale: Number of indexed documents that have
                  been modified or deleted locally (need re-indexing).
        """
        local_hashes = list((await self.get_document_hashes()).values())
        stored_hashes = list((await store.get_document_hashes(self.collection_name)).values())
        num_documents = len(stored_hashes)
        num_documents_stale = len([h for h in stored_hashes if h not in local_hashes])
        num_nodes = await store.count_nodes(self.collection_name)
        return {
            "repository": self.name,
            "path": self.path,
            "nodes_indexed": num_nodes,
            "documents_indexed": num_documents,
            "documents_indexed_stale": num_documents_stale,
        }

    async def index(self, full: bool = False) -> IndexResult:
        """
        Parse and index all code files in the repository.

        Performs intelligent incremental indexing by comparing document content
        hashes to detect changes. Only processes files that are new, modified,
        or deleted since the last indexing operation.

        The indexing process:
        1. Walks the repository to find eligible source files
        2. Computes content hashes for change detection
        3. Identifies new, modified, and deleted files
        4. Parses code into semantic nodes (functions, classes, etc.)
        5. Generates embeddings and stores nodes in the vector store
        6. Cleans up nodes for deleted files

        Files are processed according to repository settings:
        - Respects max_files limit to prevent memory issues
        - Honors ignore patterns from .gitignore and configuration
        - Skips binary, minified, and oversized files
        - Batches upsert operations for efficiency

        Args:
            full: If True, performs a full re-index by deleting the existing
                collection and re-parsing all files. If False (default),
                performs incremental indexing based on content hashes.

        Returns:
            IndexResult containing detailed statistics about the indexing
            operation, including files processed, nodes added/updated/deleted,
            and any errors encountered.
        """
        start_time = datetime.now(UTC)

        result = IndexResult()

        # Load per-repo configuration
        repo_settings = self.settings
        upsert_batch_size = repo_settings.upsert_batch_size
        max_files = repo_settings.max_files

        # On full index, recreate the collection
        if full:
            await store.delete_collection(self.collection_name)
            logger.info(f"Performing full inedex for repository: {self.name}")

        # Ensure collection exists
        await store.ensure_collection(self.collection_name)

        # Initialize walker
        walker = Walker(self)

        # Get stored document hashes for change detection
        stored_hashes = await store.get_document_hashes(self.collection_name)
        stored_paths = set(stored_hashes.keys())

        # Track what we've walked and what needs processing
        walked_paths: set[str] = set()
        files_to_process: list[dict] = []  # (doc_dict, is_new)

        # Walk the repository and identify changes
        async for doc in walker.walk():
            doc = Document.model_validate(doc)
            result.files_checked += 1
            walked_paths.add(doc.path)

            stored_hash = stored_hashes.get(doc.path)

            if stored_hash is None:
                # New file
                logger.debug(f"New file detected: {doc.path}")
                files_to_process.append({"doc": doc, "is_new": True})
            elif stored_hash != doc.hash:
                # Modified file
                logger.debug(
                    f"Modified file detected: {doc.path} "
                    f"(stored: {stored_hash[:8]}, current: {doc.hash[:8]})"
                )
                files_to_process.append({"doc": doc, "is_new": False})
            # else: unchanged, skip

        # Identify deleted files
        deleted_paths = list(stored_paths - walked_paths)

        # Respect max_files limit
        if len(files_to_process) > max_files:
            result.skipped_files = len(files_to_process) - max_files
            files_to_process = files_to_process[:max_files]
            logger.warning(
                f"Indexing limited to {max_files} files, skipping {result.skipped_files} files"
            )

        # Delete nodes for modified files (before re-adding)
        if modified_paths := [f["doc"].path for f in files_to_process if not f["is_new"]]:
            await store.delete_by_document_paths(self.collection_name, modified_paths)

        # Parse and upsert nodes in batches
        pending_nodes: list[Node] = []

        for file_info in files_to_process:
            doc = file_info["doc"]
            is_new = file_info["is_new"]

            try:
                parser = get_parser(doc.path)
                if parser is None:
                    logger.debug(f"No parser available for {doc.path}")
                    continue

                logger.info(f"Parsing {doc.path}")

                # Parse document into nodes
                file_nodes: list[Node] = []
                for content, metadata in parser.parse(doc.content):
                    node = Node(
                        content=content,
                        metadata=NodeMetadata(
                            hash=doc.hash,
                            repo_path=self.path,
                            document_path=doc.path,
                            language=metadata.get("language", "unknown"),
                            node_type=metadata.get("node_type", "unknown"),
                            node_name=metadata.get("node_name") or "",
                            start_byte=metadata.get("start_byte", 0),
                            end_byte=metadata.get("end_byte", 0),
                            start_line=metadata.get("start_line", 0),
                            end_line=metadata.get("end_line", 0),
                            documentation=metadata.get("documentation"),
                            parent_scope=metadata.get("parent_scope"),
                            signature=metadata.get("signature"),
                            extra=metadata.get("extra", {}),
                        ),
                    )
                    file_nodes.append(node)

                if not file_nodes:
                    logger.debug(f"No nodes extracted from {doc.path}")
                    placeholder_node = Node(
                        content="",  # empty content for placeholder
                        metadata=NodeMetadata(
                            hash=doc.hash,
                            repo_path=self.path,
                            document_path=doc.path,
                            language="unknown",
                            node_type="__document_marker__",  # identify placeholders
                            node_name="",
                            start_byte=0,
                            end_byte=0,
                            start_line=0,
                            end_line=0,
                        ),
                    )
                    file_nodes = [placeholder_node]

                if file_nodes:
                    pending_nodes.extend(file_nodes)

                    # Only count as indexed if it's not just a placeholder
                    if file_nodes[0].metadata.node_type != "__document_marker__":
                        result.files_indexed.append(doc.path)
                        if is_new:
                            result.nodes_added += len(file_nodes)
                        else:
                            result.nodes_updated += len(file_nodes)

                    # Batch upsert when we have enough nodes
                    if len(pending_nodes) >= upsert_batch_size:
                        await store.upsert_nodes(self.collection_name, pending_nodes)
                        pending_nodes = []

            except Exception as e:
                error_msg = f"Failed to parse {doc.path}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

        # Upsert any remaining nodes
        if pending_nodes:
            await store.upsert_nodes(self.collection_name, pending_nodes)

        # Delete nodes for removed files
        if deleted_paths:
            await store.delete_by_document_paths(self.collection_name, deleted_paths)
            result.files_deleted = deleted_paths
            # We don't know exact node count deleted, but track file count
            result.nodes_deleted = len(deleted_paths)  # Approximation

        end_time = datetime.now(UTC)

        # Finalize result
        result.indexed_at = end_time
        result.duration = (end_time - start_time).total_seconds()

        logger.debug(f"Indexing complete for {self.name}\n{result.summary}")

        return result
