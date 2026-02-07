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
        from indexter.store import VectorStore

        # Register a new repository
        repo = await Repo.init(Path("/path/to/my-project"))

        # Index all code files
        async with VectorStore() as store:
            result = await repo.index(store)
            print(f"Indexed {result.nodes_added} nodes")

            # Search for code
            results = await repo.search("database connection handling", store)
            for result in results.results:
                print(f"{result.metadata['document_path']}: {result.score}")

    Retrieve existing repositories::

        # Get a specific repository
        repo = await Repo.get_one("my-project")
        print(f"Stale: {repo.is_stale}")

        # List all repositories
        repos = await Repo.get_all()

See Also:
    - ``indexter.config``: Configuration system for global and per-repo settings
    - ``indexter.walker``: File discovery and filtering
    - ``indexter.parser``: Tree-sitter based code parsing
    - ``indexter.store``: Qdrant vector store integration
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from .cache import CacheManager
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


class Repo(BaseModel):
    """
    A git repository configured and managed by Indexter.

    Represents a Git repository that has been added to Indexter for indexing.
    Provides methods for repository management (get_one, get_all, remove_one, remove_all)
    and indexing operations (parse, search, status).

    The repository configuration includes paths, ignore patterns, and indexing
    parameters that control how files are processed and stored.
    """

    settings: RepoSettings = Field(description="Configuration settings for the repository")

    @computed_field
    @property
    def collection_name(self) -> str:
        """Name of the VectorStore collection for this repo."""
        collection_name = self.settings.collection_name
        return collection_name

    @computed_field
    @property
    def is_stale(self) -> bool:
        """
        Whether the repository index is stale compared to the current files.

        Compares the cached hashmap (from last index operation) with a freshly
        computed hashmap of current file contents. Returns True if:
        - No cached hashmap exists (never indexed)
        - The cached hashmap differs from the current hashmap (files changed)

        Returns:
            True if the index needs updating, False if up-to-date.
        """
        cached_hashmap = self._get_cached_hashmap()
        if not cached_hashmap:
            return True  # No cache means stale
        current_hashmap = self._get_hashmap()
        return cached_hashmap != current_hashmap

    @computed_field
    @property
    def path(self) -> str:
        """Absolute path to the repository root."""
        path = str(self.settings.path)
        return path

    @computed_field
    @property
    def name(self) -> str:
        """Name of the repository."""
        name = self.settings.name
        return name

    @computed_field
    @property
    def metadata(self) -> RepoMetadata:
        """
        Metadata about the repository's current state.

        Walks the repository files and parses them to compute aggregate
        metadata including document counts, node counts, languages, and
        node types. Respects the max_files setting from repository config.

        Returns:
            RepoMetadata with aggregated statistics about indexed content.
        """
        repo_metadata = RepoMetadata()
        for path, content, doc_metadata in Walker(self).walk():
            if repo_metadata.documents == self.settings.max_files:
                break
            try:
                doc = Document(
                    path=path,
                    content=content,
                    metadata=doc_metadata,
                )
                parser = Parser(doc)
                for _, meta in parser.parse():
                    if getattr(meta, "node_type", None) != "N/A":
                        node_types = set(repo_metadata.node_types)
                        node_types.add(meta.node_type)
                        repo_metadata.node_types = sorted(node_types)
                    nodes = repo_metadata.nodes or 0
                    nodes += 1
                    repo_metadata.nodes = nodes
                document_paths = set(repo_metadata.document_paths)
                document_paths.add(doc.path)
                repo_metadata.document_paths = sorted(document_paths)
                if hasattr(parser, "language"):
                    languages = set(repo_metadata.languages)
                    languages.add(str(parser.language))
                    repo_metadata.languages = sorted(languages)
                documents = repo_metadata.documents
                documents += 1
                repo_metadata.documents = documents
            except Exception:  # noqa: S112
                continue

        return repo_metadata

    @cached_property
    def cache(self) -> CacheManager:
        """
        Cache manager for repository-specific persistent data.

        Returns a CacheManager instance for storing and retrieving cached
        data like hashmaps. The cache is stored in the XDG data directory.

        Returns:
            CacheManager instance bound to this repository.
        """
        return CacheManager(self)

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
    async def get_one(cls, name: str) -> Repo:
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
    async def get_all(cls) -> list[Repo]:
        """
        Retrieve all configured repositories.

        Returns:
            List of Repo instances for all configured repositories.
        """
        repo_settings = await RepoSettings.load()
        return [cls(settings=settings) for settings in repo_settings]

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

        # Delete cache directory
        cache_dir = repo.cache.cache_dir
        if cache_dir.exists():
            for file in cache_dir.iterdir():
                file.unlink()
            cache_dir.rmdir()

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
        repos = await cls.get_all()
        if not repos:
            return False

        for repo in repos:
            logger.info(f"Removing repository: {repo.name}")
            # Remove collection from store
            await store.delete_collection(repo.collection_name)
            # Delete cache directory
            cache_dir = repo.cache.cache_dir
            if cache_dir.exists():
                for file in cache_dir.iterdir():
                    file.unlink()
                cache_dir.rmdir()
                logger.info(f"Removed repository cache: {cache_dir}")

        # Clear repos.json
        await RepoSettings.save([])
        logger.info("Removed all repositories")
        return True

    def _get_hashmap(self) -> dict[str, str]:
        """
        Build a map of document hashes by document path.

        Walks the repository and computes a hash for each document.
        The hashmap maps document paths to document hashes, enabling
        efficient change detection.

        Respects max_files setting and logs a warning for files that
        fail to process.

        Returns:
            Dict mapping document paths to document hashes.
        """
        count = 0
        skipped = 0

        def _max_files_limit_reached() -> bool:
            nonlocal count
            nonlocal skipped
            if count >= self.settings.max_files:
                skipped += 1
                return True
            return False

        hashmap: dict[str, str] = {}
        for path, content, metadata in Walker(self).walk():
            if _max_files_limit_reached():
                logger.warning(f"Reached max_files limit ({self.settings.max_files}): {path} will be skipped.")
                break
            try:
                doc = Document(
                    path=path,
                    content=content,
                    metadata=metadata,
                )
                hashmap[doc.path] = doc.hash
                count += 1
            except Exception as e:
                logger.warning(f"Failed to add {path} to hashmap: {e}")
                continue

        if skipped > 0:
            logger.warning(f"Skipped {skipped} files due to max_files limit.")

        return hashmap

    def _get_cached_hashmap(self) -> dict[str, str]:
        """
        Retrieve the cached hashmap from the last index operation.

        Returns:
            Dict mapping document paths to document hashes, or empty dict if
            no cached hashmap exists.
        """
        data = self.cache.get("hashmap")
        if not data:
            return {}
        hashmap = json.loads(data)
        return hashmap

    def _set_hashmap(self, hashmap: dict[str, str]) -> None:
        """
        Cache the hashmap for future change detection.

        Persists the hashmap to the repository's cache directory. Called
        after a successful index operation to enable incremental updates.

        Args:
            hashmap: Dict mapping document paths to document hashes.
        """
        data = json.dumps(hashmap)
        self.cache.set("hashmap", data)

    def _delete_hashmap(self) -> bool:
        """
        Delete the cached hashmap.

        Removes the cached hashmap from the repository's cache directory.
        Used when performing a full re-index to clear stale data.

        Returns:
            True if the hashmap was deleted, False if it did not exist.
        """
        return self.cache.delete("hashmap")

    def _get_hashes(self, hashmap: dict[str, str]) -> list[str]:
        """
        A list of all document hashes.

        Args:
            hashmap: Dict mapping document paths to document hashes.

        Returns:
            List of all document hashes.
        """
        return list(hashmap.values())

    def _get_hashes_to_add(self, hashes: list[str], cached_hashes: list[str]) -> list[str]:
        """
        Determine which document hashes are new or modified.

        Computes the set difference between current and cached hashes to
        identify documents that need to be added to the vector store.

        Args:
            hashes: Current document hashes from the repository.
            cached_hashes: Previously cached document hashes.
        Returns:
            List of hashes present in current state but not in cached state.
        """
        return list(set(hashes) - set(cached_hashes or []))

    def _get_hashes_to_delete(self, hashes: list[str], cached_hashes: list[str]) -> list[str]:
        """
        Determine which documents have been removed or modified.

        Computes the set difference to identify document hashes that existed in the
        cached state but are no longer present in the current state.

        Args:
            hashes: Current document hashes from the repository.
            cached_hashes: Previously cached document hashes.

        Returns:
            List of hashes present in cached state but not in current state.
        """
        return list(set(cached_hashes or []) - set(hashes))

    async def index(self, store: VectorStore, full: bool = False) -> IndexResult:
        """
        Parse and index files in the repository.

        By default, performs intelligent incremental indexing using hash
        comparison to detect changes at the document level. Identifies new,
        modified, and deleted documents by comparing document hashes, and
        only re-parses what changed. When full=True, re-indexes all files
        by recreating the collection.

        The indexing process:
        1. Builds a hashmap from current repository state (document path → document hash)
        2. Compares with cached hashmap to identify changed documents
        3. Deletes nodes belonging to removed/modified documents
        4. Parses changed documents and upserts their nodes
        5. Saves the new hashmap to cache

        Files are processed according to repository settings:
        - Respects ignore patterns from .gitignore and configuration
        - Skips binary, minified, and oversized files
        - Limits indexing to max_files (additional files skipped with warning)
        - Batches upsert operations for efficiency

        Args:
            store: VectorStore instance for storing embeddings.
            full: If True, performs a full re-index by deleting the existing
                collection and re-parsing all files. If False (default),
                performs incremental indexing based on hash comparison.

        Returns:
            IndexResult containing detailed statistics about the indexing
            operation, including documents indexed/deleted, nodes added/deleted,
            and any errors encountered.
        """
        start_time = datetime.now(UTC)

        # Load per-repo configuration
        repo_settings = self.settings
        upsert_batch_size = repo_settings.upsert_batch_size

        # Ensure collection exists
        await store.ensure_collection(self.collection_name)

        # On full index, recreate the collection and clear cache
        if full:
            await store.delete_collection(self.collection_name)
            self._delete_hashmap()
            logger.info(f"Performing full index for repository: {self.name}")

        # Initialize result
        result = IndexResult(repo=self.name, repo_path=self.path)

        # Quick check: if hashmaps match, no changes
        hashmap = self._get_hashmap()
        cached_hashmap = self._get_cached_hashmap()
        if cached_hashmap and cached_hashmap == hashmap:
            logger.info(f"No changes detected for {self.name}")
            end_time = datetime.now(UTC)
            result.indexed_at = end_time
            result.duration = (end_time - start_time).total_seconds()
            return result

        # Retrieve hashes for current and cached state
        hashes = self._get_hashes(hashmap)
        cached_hashes = self._get_hashes(cached_hashmap or {})

        # Delete nodes for removed/modified documents by hash
        if hashes_to_delete := self._get_hashes_to_delete(hashes, cached_hashes):
            logger.info(f"Deleting {len(hashes_to_delete)} nodes from removed/modified documents")
            # Delete nodes by document hash
            nodes_deleted = await store.delete_by_hashes(self.collection_name, hashes_to_delete)
            result.nodes_deleted = nodes_deleted
            # Identify which documents were deleted based on hashes
            documents_deleted = [path for path, hash in (cached_hashmap or {}).items() if hash in hashes_to_delete]
            result.documents_deleted = documents_deleted

        # Parse and upsert nodes for new/modified documents and nodes
        if hashes_to_add := self._get_hashes_to_add(hashes, cached_hashes):
            nodes_to_upsert: list[Node] = []
            excluded_paths = [path for path, hash in hashmap.items() if hash not in hashes_to_add]
            for path, content, metadata in Walker(self).walk(excluded_paths=excluded_paths):
                try:
                    logger.info(f"Parsing document: {path}")
                    nodes: list[Node] = []
                    doc = Document(path=path, content=content, metadata=metadata)
                    for node_content, node_metadata in Parser(doc).parse():
                        node = Node.from_parsed(
                            content=node_content,
                            metadata=node_metadata,
                        )
                        nodes.append(node)
                    nodes_to_upsert.extend(nodes)
                    result.documents_indexed.append(doc.path)
                except Exception as e:
                    error_msg = f"Failed to parse {doc.path}: {e}"
                    result.errors.append(error_msg)
                    logger.warning(error_msg)
                    continue

                # Batch upsert when we have enough nodes
                if len(nodes_to_upsert) >= upsert_batch_size:
                    await store.upsert_nodes(self.collection_name, nodes_to_upsert)
                    result.nodes_added += len(nodes_to_upsert)
                    nodes_to_upsert = []

            # Upsert any remaining nodes
            if nodes_to_upsert:
                await store.upsert_nodes(self.collection_name, nodes_to_upsert)
                result.nodes_added += len(nodes_to_upsert)
                nodes_to_upsert = []

        # Finalize result
        end_time = datetime.now(UTC)
        result.indexed_at = end_time
        result.duration = (end_time - start_time).total_seconds()

        # Save the new hashmap to cache
        self._set_hashmap(hashmap)

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
        search_results.repo = self.name
        search_results.repo_path = self.path
        return search_results
