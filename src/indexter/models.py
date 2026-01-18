"""
Core domain models for Indexter repository management.

This module defines the primary models for managing indexed code repositories:
``Repo`` for repository operations and ``RepoMetadata`` for status information.

The ``Repo`` model serves as the main entry point for all repository operations,
including initialization, indexing, searching, and removal. It coordinates between
the walker (file discovery), parser (code analysis), and store (vector database)
components to provide a unified API.

Architecture:
    The module follows an async-first design pattern. All I/O operations
    (file walking, parsing, vector store access) are asynchronous to support
    efficient processing of large repositories.

    Repository lifecycle:
        1. ``Repo.init(path)`` - Register a new repository
        2. ``repo.index()`` - Parse and embed code into vector store
        3. ``repo.search(query)`` - Semantic search over indexed code
        4. ``Repo.remove_one(name)`` - Remove repository and its data

    Change detection:
        Incremental indexing uses content hashes (SHA-256 of path + content)
        to detect new, modified, and deleted files. Only changed files are
        re-processed on subsequent index operations.

Classes:
    RepoMetadata: Status information for an indexed repository including
        document counts, languages, node types, and staleness indicators.
    Repo: Main repository model with methods for initialization, indexing,
        searching, and management operations.

Example:
    Initialize and index a repository::

        from pathlib import Path
        from indexter import Repo

        # Register a new repository
        repo = await Repo.init(Path("/path/to/my-project"))

        # Index all code files
        result = await repo.index()
        print(f"Indexed {result.nodes_added} nodes")

        # Search for code
        results = await repo.search("database connection handling")
        for result in results.results:
            print(f"{result.metadata['document_path']}: {result.score}")

    Retrieve existing repositories::

        # Get a specific repository
        repo = await Repo.get_one("my-project", with_metadata=True)
        print(f"Stale: {repo.metadata.is_stale}")

        # List all repositories
        repos = await Repo.get_all()

See Also:
    - ``indexter.config``: Configuration system for global and per-repo settings
    - ``indexter.walker``: File discovery and filtering
    - ``indexter.parser``: Tree-sitter based code parsing
    - ``indexter.store``: Qdrant vector store integration
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from .config import RepoSettings
from .exceptions import RepoExistsError, RepoNotFoundError
from .parser import Parser
from .parser.models import Node
from .store import VectorStore
from .store.models import IndexResult, SearchResults
from .walker import Walker
from .walker.models import Document

logger = logging.getLogger(__name__)


class RepoMetadata(BaseModel):
    """
    Status information for an indexed repository.

    Provides current state of repository indexing including node counts,
    document counts, and staleness indicators.
    """

    document_paths: list[str] = Field(description="List of indexed document paths (relative to Repo root)")
    is_stale: bool = Field(description="Whether the repository index is stale")
    languages: list[str] = Field(description="Indexed languages")
    node_types: list[str] = Field(description="Indexed node types")
    nodes_indexed: int = Field(description="Total number of nodes indexed")

    @computed_field
    @property
    def documents_indexed(self) -> int:
        """Number of indexed documents."""
        return len(self.document_paths)

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

    @classmethod
    async def from_repo(cls, repo: Repo, store: VectorStore) -> RepoMetadata:
        """
        Create RepoMetadata by querying the repository's current status.

        Args:
            repo: Repo instance to get metadata for.
            store: VectorStore instance for querying indexed data.

        Returns:
            RepoMetadata model with current repository statistics.
        """
        _document_paths: set[str] = set()
        _languages: set[str] = set()
        _node_types: set[str] = set()

        _local_hashes: dict[str, str] = {}
        _stored_hashes: dict[str, str] = await store.get_document_hashes(repo.collection_name)

        walker = Walker(repo)
        async for path, content, metadata in walker.walk():
            doc = Document(
                path=path,
                content=content,
                metadata=metadata,
            )
            _document_paths.add(doc.path)
            _local_hashes[doc.path] = doc.metadata.hash
            parser = Parser(doc)
            if not hasattr(parser, "language"):
                continue
            _languages.add(str(parser.language))
            try:
                for _, meta in parser.parse():
                    if getattr(meta, "node_type", None) != "N/A":
                        _node_types.add(meta.node_type)
            except Exception as e:
                logger.warning(f"Failed to parse {doc.path} for metadata extraction: {e}")

        document_paths = list(_document_paths)
        languages = list(_languages)
        node_types = list(_node_types)
        nodes_indexed = await store.count_nodes(repo.collection_name)
        is_stale = _local_hashes != _stored_hashes

        return RepoMetadata(
            document_paths=document_paths,
            languages=languages,
            node_types=node_types,
            nodes_indexed=nodes_indexed,
            is_stale=is_stale,
        )


class Repo(BaseModel):
    """
    A git repository configured and managed by Indexter.

    Represents a Git repository that has been added to Indexter for indexing.
    Provides methods for repository management (get_one, get_all, remove_one, remove_all)
    and indexing operations (parse, search, status).

    The repository configuration includes paths, ignore patterns, and indexing
    parameters that control how files are processed and stored.
    """

    metadata: RepoMetadata | None = Field(default=None, description="Metadata about the repository")
    settings: RepoSettings = Field(description="Configuration settings for the repository")

    # Properties derived from settings for convenience

    @computed_field
    @property
    def collection_name(self) -> str:
        """Name of the VectorStore collection for this repo."""
        collection_name = self.settings.collection_name
        return collection_name

    @computed_field
    @property
    def name(self) -> str:
        """Name of the repository."""
        name = self.settings.name
        return name

    @computed_field
    @property
    def path(self) -> str:
        """Absolute path to the repository root."""
        path = str(self.settings.path)
        return path

    @classmethod
    async def init(cls, path: Path) -> Repo:
        """
        Initialize and register a new repository with Indexter.

        Validates the path contains a .git directory, checks for name conflicts
        with existing repositories, and adds the repository to the configuration.
        If the repository is already configured at the same path, returns the
        existing Repo instance without modification.

        Repository names are automatically derived from the directory name.

        Args:
            path: Path to the git repository root directory.

        Returns:
            Repo instance for the initialized repository.

        Raises:
            ValueError: If the path does not contain a .git directory.
            RepoExistsError: If a different repository with the same derived name
                already exists at a different path.
        """
        repo_settings = await RepoSettings.load()
        resolved_path = path.resolve()

        # Create new config to get the derived name
        settings = RepoSettings(path=resolved_path)

        # Check if name already exists
        for existing in repo_settings:
            if existing.name == settings.name:
                if existing.path.resolve() == resolved_path:
                    # Same repo, already configured
                    logger.info(f"Repository already configured: {settings.name}")
                    return cls(settings=existing)
                else:
                    # Different repo with same name
                    raise RepoExistsError(
                        f"A repository named '{existing.name}' already exists "
                        f"at {existing.path}. Rename the directory to use a unique name."
                    )

        repo_settings.append(settings)
        await RepoSettings.save(repo_settings)

        logger.info(f"Added repository: {settings.name} ({resolved_path})")
        return cls(settings=settings)

    @classmethod
    async def get_one(cls, name: str, store: VectorStore | None = None, with_metadata: bool = False) -> Repo:
        """
        Retrieve a configured repository by name.

        Searches the configuration for a repository matching the given name
        and returns the corresponding Repo instance.

        Args:
            name: Repository name (derived from the directory name containing .git).
            store: VectorStore instance for querying metadata. Required if with_metadata is True.
            with_metadata: If True, populate the metadata attribute. Requires store parameter.

        Returns:
            Repo instance for the requested repository. If with_metadata is True,
            the Repo instance will have its metadata attribute populated.

        Raises:
            RepoNotFoundError: If no repository with the given name is configured.
            ValueError: If with_metadata is True but store is not provided.
        """
        if with_metadata and store is None:
            raise ValueError("store parameter is required when with_metadata is True")
        repos = await RepoSettings.load()
        for repo_settings in repos:
            if repo_settings.name == name:
                repo = cls(settings=repo_settings)
                if with_metadata and store is not None:
                    repo.metadata = await RepoMetadata.from_repo(repo, store)
                return repo
        raise RepoNotFoundError(f"Repository not found: {name}")

    @classmethod
    async def get_all(cls, store: VectorStore | None = None, with_metadata: bool = False) -> list[Repo]:
        """
        Retrieve all configured repositories.

        Args:
            store: VectorStore instance for querying metadata. Required if with_metadata is True.
            with_metadata: If True, populate the metadata attribute. Requires store parameter.

        Returns:
            List of Repo instances for all configured repositories. If with_metadata is True,
            each Repo instance will have its metadata attribute populated.

        Raises:
            ValueError: If with_metadata is True but store is not provided.
        """
        if with_metadata and store is None:
            raise ValueError("store parameter is required when with_metadata is True")
        repo_settings = await RepoSettings.load()
        repos = [cls(settings=settings) for settings in repo_settings]
        if with_metadata and store is not None:
            for repo in repos:
                repo.metadata = await RepoMetadata.from_repo(repo, store)
        return repos

    @classmethod
    async def remove_one(cls, name: str, store: VectorStore) -> bool:
        """
        Remove a repository and its indexed data.

        Deletes the repository's vector store collection and removes it from
        the configuration. This operation is permanent and cannot be undone.

        Args:
            name: Name of the repository to remove.
            store: VectorStore instance for deleting the collection.

        Returns:
            True if the repository was successfully removed, False if it was
            already removed by another process (race condition).

        Raises:
            RepoNotFoundError: If no repository with the given name exists.
        """
        repo = await cls.get_one(name)

        # Delete collection from store
        await store.delete_collection(repo.collection_name)

        # Remove from repos.json
        repo_settings = await RepoSettings.load()
        new_repo_settings = [r for r in repo_settings if r.name != name]
        await RepoSettings.save(new_repo_settings)
        if new_repo_settings != repo_settings:
            logger.info(f"Removed repository: {name}")
            return True
        return False

    @classmethod
    async def remove_all(cls, store: VectorStore) -> bool:
        """
        Remove all configured repositories and their indexed data.

        Deletes all repositories' vector store collections and clears the
        configuration. This operation is permanent and cannot be undone.

        Args:
            store: VectorStore instance for deleting collections.

        Returns:
            True if any repositories were removed, False if there were none.
        """
        repo_settings = await RepoSettings.load()
        if not repo_settings:
            return False

        for settings in repo_settings:
            collection_name = settings.collection_name
            await store.delete_collection(collection_name)
            logger.info(f"Removed repository collection: {collection_name}")

        # Clear repos.json
        await RepoSettings.save([])
        logger.info("Removed all repositories")
        return True

    async def index(self, store: VectorStore, full: bool = False) -> IndexResult:
        """
        Parse and index files in the repository.

        By default, performs intelligent incremental indexing by comparing document
        content hashes to detect changes. Only processes files that are new, modified,
        or deleted since the last indexing operation. When full=True, re-indexes all
        files by recreating the collection.

        The indexing process:
        1. Walks the repository to find eligible source files
        2. Computes content hashes for change detection
        3. Identifies new, modified, and deleted files
        4. Parses code into semantic nodes (functions, classes, methods, etc.)
        5. Creates placeholder nodes for files with no parseable code
        6. Generates embeddings and stores nodes in the vector store
        7. Deletes nodes for removed files

        Files are processed according to repository settings:
        - Honors ignore patterns from .gitignore and configuration
        - Skips binary, minified, and oversized files
        - Limits indexing to max_files (additional files skipped with warning)
        - Batches upsert operations for efficiency

        Args:
            store: VectorStore instance for storing embeddings.
            full: If True, performs a full re-index by deleting the existing
                collection and re-parsing all files. If False (default),
                performs incremental indexing based on content hashes.

        Returns:
            IndexResult containing detailed statistics about the indexing
            operation, including files processed, nodes added/updated/deleted,
            and any errors encountered.
        """
        start_time = datetime.now(UTC)

        result = IndexResult(repo=self.name, repo_path=self.path)

        # Load per-repo configuration
        repo_settings = self.settings
        upsert_batch_size = repo_settings.upsert_batch_size
        max_files = repo_settings.max_files

        # On full index, recreate the collection
        if full:
            await store.delete_collection(self.collection_name)
            logger.info(f"Performing full index for repository: {self.name}")

        # Ensure collection exists
        await store.ensure_collection(self.collection_name)

        # Get stored document hashes for change detection
        stored_hashes = await store.get_document_hashes(self.collection_name)
        stored_paths = set(stored_hashes.keys())

        # Track walked paths for detecting deletions
        walked_paths: set[str] = set()

        # Parse and upsert nodes in batches - single pass, streaming
        pending_nodes: list[Node] = []
        files_indexed = 0

        walker = Walker(self)
        async for path, content, metadata in walker.walk():
            result.documents_checked += 1
            walked_paths.add(path)

            # Check if file needs indexing
            stored_hash = stored_hashes.get(path)
            is_new = stored_hash is None
            is_modified = stored_hash is not None and stored_hash != metadata.hash

            if not is_new and not is_modified:
                # Unchanged file, skip
                continue

            # Respect max_files limit
            if files_indexed >= max_files:
                result.skipped_documents += 1
                if result.skipped_documents == 1:
                    # Log warning once when limit is first exceeded
                    logger.warning(f"Indexing limited to {max_files} files, additional files will be skipped")
                continue

            files_indexed += 1

            if is_modified:
                # stored_hash is guaranteed to be not None here due to is_modified check
                if stored_hash is None:
                    raise RuntimeError(f"Unexpected None hash for modified file: {path}")
                logger.debug(
                    f"Modified file detected: {path} (stored: {stored_hash[:8]}, current: {metadata.hash[:8]})"
                )
                # Delete old nodes for this file before re-adding
                await store.delete_by_document_paths(self.collection_name, [path])
            else:
                logger.debug(f"New file detected: {path}")

            # Process file immediately
            doc = Document(path=path, content=content, metadata=metadata)

            # Parse the file into nodes
            try:
                logger.info(f"Parsing {doc.path}")
                doc_nodes: list[Node] = []
                parser = Parser(doc)
                for node_content, node_metadata in parser.parse():
                    node = Node(content=node_content, metadata=node_metadata)
                    doc_nodes.append(node)
            except Exception as e:
                error_msg = f"Failed to parse {doc.path}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                continue

            if not doc_nodes:
                logger.debug(f"No nodes extracted from {doc.path}")
                placeholder_node = Node.placeholder(self.name, self.path, doc.path, doc.metadata.hash)
                doc_nodes = [placeholder_node]

            pending_nodes.extend(doc_nodes)

            # Only count as indexed if it's not just a placeholder
            if doc_nodes[0].metadata.node_type != "__PLACEHOLDER__":
                result.documents_indexed.append(doc.path)
                if is_new:
                    result.nodes_added += len(doc_nodes)
                else:
                    result.nodes_updated += len(doc_nodes)

            # Batch upsert when we have enough nodes
            if len(pending_nodes) >= upsert_batch_size:
                await store.upsert_nodes(self.collection_name, pending_nodes)
                pending_nodes = []

        # Upsert any remaining nodes
        if pending_nodes:
            await store.upsert_nodes(self.collection_name, pending_nodes)

        # Identify and delete nodes for removed files
        deleted_paths = list(stored_paths - walked_paths)
        if deleted_paths:
            await store.delete_by_document_paths(self.collection_name, deleted_paths)
            result.documents_deleted = deleted_paths

        end_time = datetime.now(UTC)

        # Finalize result
        result.indexed_at = end_time
        result.duration = (end_time - start_time).total_seconds()

        logger.debug(f"Indexing complete for {self.name}\n{result.summary}")

        return result

    async def search(
        self,
        query: str,
        store: VectorStore,
        document_path: str | None = None,
        language: str | None = None,
        node_type: str | None = None,
        node_name: str | None = None,
        parent_scope: str | None = None,
        has_documentation: bool | None = None,
        limit: int | None = None,
    ) -> SearchResults:
        """
        Perform semantic search over indexed code nodes in the repository.

        Searches the repository's vector store using embedding-based similarity.
        Results can be filtered by multiple metadata criteria to narrow down
        the search scope.

        Args:
            query: Natural language or code search query.
            store: VectorStore instance for searching.
            document_path: Filter by document path. Use exact match or prefix with
                trailing '/' for directory filtering.
            language: Filter by programming language (e.g., 'python', 'javascript').
            node_type: Filter by code construct type (e.g., 'function', 'class', 'method').
            node_name: Filter by exact node name (function/class name).
            parent_scope: Filter by the enclosing scope or class name (e.g., 'MyClass' for methods).
            has_documentation: Filter by documentation presence. True for nodes
                with docstrings/comments, False for undocumented nodes.
            limit: Maximum number of results to return. Defaults to the repository's top_k setting.

        Returns:
            SearchResults model containing matched code chunks with scores, metadata,
            and query context. Ordered by relevance (highest score first).
        """

        search_results = await store.search(
            collection_name=self.collection_name,
            query=query,
            limit=limit or self.settings.top_k,
            document_path=document_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            parent_scope=parent_scope,
            has_documentation=has_documentation,
        )

        # assign repo info to results
        search_results.repo = self.name
        search_results.repo_path = self.path

        return search_results
