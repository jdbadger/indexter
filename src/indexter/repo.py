"""Repository management and indexing operations.

This module contains the ``Repo`` class — the primary interface for registering,
indexing, and searching Git repositories — along with the helper managers that
handle caching and vector-store interactions on its behalf.

Classes:
    CacheManager:
        File-based key/value cache scoped to a single repository. Stores
        the document hashmap used for incremental change detection in
        ``~/.cache/indexter/<repo>/``.

    StoreManager:
        Thin wrapper around ``QdrantClient`` scoped to a single repository's
        collection. Handles collection creation, node upsert/delete, and
        hybrid search (dense + sparse with Reciprocal Rank Fusion).

    Repo:
        Represents a registered Git repository. Class methods manage the
        repository lifecycle (``init``, ``get_one``, ``get_all``,
        ``remove_one``, ``remove_all``). Instance methods perform indexing
        and search operations.

Indexing Strategy:
    By default, ``Repo.index`` performs incremental indexing:

    1. Walk the repository and compute a hashmap (path → SHA-256).
    2. Compare with the cached hashmap from the previous run.
    3. Delete vector-store nodes for removed/modified documents.
    4. Parse changed documents and upsert new nodes.
    5. Cache the updated hashmap.

    Pass ``full=True`` to drop the collection and re-index from scratch.

Example:
    Register a repository, index it, and search::

        from pathlib import Path
        from qdrant_client import QdrantClient
        from indexter import Repo

        client = QdrantClient(host="localhost", port=6333)

        repo = Repo.init(Path("/home/user/my-project"))
        result = repo.index(client)
        print(result.summary)

        results = repo.search(client, "authentication handler", limit=5)
        for r in results.results:
            print(f"{r.score:.3f}: {r.metadata['node_name']}")

See Also:
    ``indexter.models``: Domain models (Document, Node, IndexResult, etc.)
    ``indexter.config``: Settings and RepoSettings
    ``indexter.parser``: Tree-sitter based code parsing
    ``indexter.walker``: File-system traversal with filtering
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from .config import RepoSettings, settings
from .exceptions import RepoExistsError, RepoNotFoundError
from .models import Document, IndexResult, Node, RepoMetadata, SearchResult, SearchResults
from .parser import Parser
from .walker import Walker

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cached data for a repository."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        self.cache_dir = settings.cache_dir / self.repo.name
        self.cache_key_prefix = f"{self.repo.name}"

    def _key_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{self.cache_key_prefix}_{key}.json"

    def get(self, key: str) -> str | None:
        """Get cached data by key."""
        path = self._key_path(key)
        if not path.exists():
            return None
        return path.read_text()

    def set(self, key: str, data: str) -> None:
        """Set cached data by key."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._key_path(key).write_text(data)

    def delete(self, key: str) -> bool:
        """Delete cached data by key. Returns True if deleted."""
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all cached data for this repository."""
        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                file.unlink()
            self.cache_dir.rmdir()


class StoreManager:
    def __init__(self, repo: Repo):
        self.repo = repo

    @property
    def dense_model_name(self) -> str:
        """Dense embedding model name from global settings."""
        return settings.store.embedding_model

    @property
    def sparse_model_name(self) -> str:
        """Sparse embedding model name from global settings."""
        return settings.store.sparse_embedding_model

    @property
    def dense_vector_name(self) -> str:
        """Named vector key for dense embeddings in the Qdrant collection."""
        return settings.store.embedding_model

    @property
    def sparse_vector_name(self) -> str:
        """Named vector key for sparse embeddings in the Qdrant collection."""
        return settings.store.sparse_embedding_model

    def create_collection(self, client: QdrantClient) -> None:
        """Create a collection in the vector store using fastembed vector params.

        Args:
            collection_name: Name of the collection to create.
        """
        collection_name = self.repo.collection_name
        collection_exists = client.collection_exists(collection_name)
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.dense_vector_name: models.VectorParams(
                        size=client.get_embedding_size(self.dense_model_name), distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={self.sparse_vector_name: models.SparseVectorParams()},
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")

    def delete_collection(self, client: QdrantClient) -> None:
        """Drop a collection from the vector store.

        Args:
            collection_name: Name of the collection to drop.
        """
        collection_name = self.repo.collection_name
        client.delete_collection(collection_name=collection_name)

    def ensure_collection(self, client: QdrantClient) -> bool:
        """Ensure a collection exists, creating it if necessary.

        Uses an in-memory cache to avoid repeated checks.

        Args:
            collection_name: Name of the collection to ensure exists.
        """
        collection_name = self.repo.collection_name

        # Check if collection exists
        collections = client.get_collections()
        existing_names = {c.name for c in collections.collections}

        # Create collection if it doesn't exist
        if collection_name not in existing_names:
            self.create_collection(client)
            return True
        return False

    def upsert_nodes(
        self,
        client: QdrantClient,
        nodes: list[Node],
        batch_size: int = 10,
    ) -> int:
        """Upsert nodes to a collection using fastembed for embeddings.

        Processes nodes in small sub-batches to reduce memory pressure during
        embedding generation. This is important because FastEmbed loads models
        into memory and generates embeddings for all texts at once.

        Args:
            client: AsyncQdrantClient instance to use for the operation.
            nodes: List of Node objects to upsert.
            batch_size: Number of nodes to process in each sub-batch. Smaller
                values reduce memory usage but increase API calls. Default is 10.

        Returns:
            Number of nodes upserted.
        """
        if not nodes:
            return 0

        collection_name = self.repo.collection_name

        total_upserted: int = 0
        self.ensure_collection(client)

        # Process nodes in small sub-batches to reduce memory pressure
        _batch_size = self.repo.settings.upsert_batch_size or batch_size
        for i in range(0, len(nodes), _batch_size):
            batch_nodes = nodes[i : i + _batch_size]

            # Prepare documents and metadata for this batch only
            texts = [node.content for node in batch_nodes]
            payloads = [node.as_payload() for node in batch_nodes]
            ids = [node.id for node in batch_nodes]

            # Build points with Document for automatic embedding inference
            points = [
                models.PointStruct(
                    id=id,
                    vector={
                        self.dense_vector_name: models.Document(text=text, model=self.dense_model_name),
                        self.sparse_vector_name: models.Document(text=text, model=self.sparse_model_name),
                    },
                    payload=payload,
                )
                for id, text, payload in zip(ids, texts, payloads, strict=True)
            ]

            client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_upserted += len(batch_nodes)

        return total_upserted

    def delete_nodes(self, client: QdrantClient, hashes: list[str]) -> int:
        """Delete all nodes with the provided document hashes.

        Args:
            collection_name: Name of the collection to delete from.
            hashes: List of document hashes to delete nodes for.

        Returns:
            Number of points deleted.
        """
        if not hashes:
            return 0

        self.ensure_collection(client)

        collection_name = self.repo.collection_name

        hash_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="document_hash",
                    match=models.MatchValue(value=hash_value),
                )
                for hash_value in hashes
            ]
        )

        # Count matching points before deletion (Qdrant's delete doesn't return a count)
        count_result = client.count(
            collection_name=collection_name,
            count_filter=hash_filter,
            exact=True,
        )

        # Delete using filter on document_hash
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=hash_filter),
        )

        return count_result.count

    @staticmethod
    def _build_filter_conditions(
        document_path: str | None = None,
        language: str | None = None,
        node_type: str | None = None,
        node_name: str | None = None,
        parent_scope: str | None = None,
        has_documentation: bool | None = None,
    ) -> list[models.FieldCondition]:
        """Helper method to build filter conditions for search based on provided criteria."""
        filter_conditions = []

        if document_path:
            # Support both exact match and prefix matching
            if document_path.endswith("/"):
                # Prefix match for directories
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_path",
                        match=models.MatchText(text=document_path),
                    )
                )
            else:
                # Exact match for files
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_path",
                        match=models.MatchValue(value=document_path),
                    )
                )

        if language:
            filter_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        if node_type:
            filter_conditions.append(
                models.FieldCondition(
                    key="node_type",
                    match=models.MatchValue(value=node_type),
                )
            )

        if node_name:
            filter_conditions.append(
                models.FieldCondition(
                    key="node_name",
                    match=models.MatchValue(value=node_name),
                )
            )

        if parent_scope:
            filter_conditions.append(
                models.FieldCondition(
                    key="parent_scope",
                    match=models.MatchValue(value=parent_scope),
                )
            )

        if has_documentation is not None:
            # Check if documentation field is non-empty
            if has_documentation:
                filter_conditions.append(
                    models.FieldCondition(
                        key="documentation",
                        match=models.MatchExcept.model_validate({"except": [""]}),
                    )
                )
            else:
                filter_conditions.append(
                    models.FieldCondition(
                        key="documentation",
                        match=models.MatchValue(value=""),
                    )
                )

        return filter_conditions

    def search(
        self,
        client: QdrantClient,
        query: str,
        document_path: str | None = None,
        language: str | None = None,
        node_type: str | None = None,
        node_name: str | None = None,
        parent_scope: str | None = None,
        has_documentation: bool | None = None,
        limit: int = 10,
    ) -> SearchResults:
        """Perform semantic search on a collection with optional filters.

        Args:
            collection_name: Name of the collection to search.
            query: Search query text.
            limit: Maximum number of results to return.
            document_path: Filter by document path (exact match or prefix).
            language: Filter by programming language.
            node_type: Filter by node type (e.g., 'function', 'class').
            node_name: Filter by node name (exact match).
            parent_scope: Filter by the enclosing scope or class name (e.g., 'MyClass' for methods).
            has_documentation: Filter by documentation presence (e.g. docstring or doc comments).
            limit: Maximum number of results to return (default: 10).

        Returns:
            List of SearchResult objects.
        """
        collection_name = self.repo.collection_name

        collection_was_created = self.ensure_collection(client)
        if collection_was_created:
            # Collection was just created from scratch – it's empty.
            logger.info(f"Collection {collection_name} is empty (just created).")
            return SearchResults(results=[], query=query, filters={})

        # Build filter conditions
        filter_conditions = self._build_filter_conditions(
            document_path=document_path,
            language=language,
            node_type=node_type,
            node_name=node_name,
            parent_scope=parent_scope,
            has_documentation=has_documentation,
        )

        # Build query filter
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)  # type: ignore[arg-type, ty:invalid-argument-type]

        if self.dense_vector_name is None or self.dense_model_name is None:
            raise RuntimeError("Vector store not properly initialized")

        if self.sparse_vector_name is None or self.sparse_model_name is None:
            raise RuntimeError("Vector store not properly initialized")

        prefetch = [
            models.Prefetch(
                query=models.Document(text=query, model=self.dense_model_name),
                using=self.dense_vector_name,
                limit=limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=self.sparse_model_name),
                using=self.sparse_vector_name,
                limit=limit,
            ),
        ]

        # Perform search using query_points with Document for embedding inference
        results = client.query_points(
            collection_name=collection_name,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            prefetch=prefetch,
            limit=limit,
        )

        search_results_list = []
        for point in results.points:
            content = point.payload.pop("content", "") if point.payload else ""
            score = point.score or 0.0
            metadata = point.payload or {}
            result = SearchResult(
                content=content,
                score=score,
                metadata=metadata,
            )
            search_results_list.append(result)

        filters = {
            "document_path": document_path,
            "language": language,
            "node_type": node_type,
            "node_name": node_name,
            "parent_scope": parent_scope,
            "has_documentation": has_documentation,
        }

        return SearchResults(
            results=search_results_list,
            query=query,
            filters=filters or {},
        )


class Repo:
    """
    A git repository configured and managed by Indexter.

    Represents a Git repository that has been added to Indexter for indexing.
    Provides methods for repository management (get_one, get_all, remove_one, remove_all)
    and indexing operations (parse, search, status).

    The repository configuration includes paths, ignore patterns, and indexing
    parameters that control how files are processed and stored.
    """

    def __init__(self, settings: RepoSettings):
        self.settings = settings

    @property
    def collection_name(self) -> str:
        """Name of the VectorStore collection for this repo."""
        collection_name = self.settings.collection_name
        return collection_name

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

    @property
    def path(self) -> str:
        """Absolute path to the repository root."""
        path = str(self.settings.path)
        return path

    @property
    def name(self) -> str:
        """Name of the repository."""
        name = self.settings.name
        return name

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

    @property
    def cache(self) -> CacheManager:
        """
        Cache manager for repository-specific persistent data.

        Returns a CacheManager instance for storing and retrieving cached
        data like hashmaps. The cache is stored in the XDG data directory.

        Returns:
            CacheManager instance bound to this repository.
        """
        return CacheManager(self)

    @property
    def store(self) -> StoreManager:
        """
        Store manager for repository-specific vector store operations.

        Returns a StoreManager instance for performing vector store operations
        like upsert, search, and delete.

        Returns:
            StoreManager instance bound to this repository.
        """
        return StoreManager(self)

    @classmethod
    def init(cls, path: Path) -> Repo:
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
        repo_settings = RepoSettings.load()
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
        RepoSettings.save(repo_settings)

        logger.info(f"Added repository: {settings.name} ({resolved_path})")
        return cls(settings=settings)

    @classmethod
    def get_one(cls, name: str) -> Repo:
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
        repos = RepoSettings.load()
        for repo_settings in repos:
            if repo_settings.name == name:
                return cls(settings=repo_settings)
        raise RepoNotFoundError(f"Repository not found: {name}")

    @classmethod
    def get_all(cls) -> list[Repo]:
        """
        Retrieve all configured repositories.

        Returns:
            List of Repo instances for all configured repositories.
        """
        repo_settings = RepoSettings.load()
        return [cls(settings=settings) for settings in repo_settings]

    @classmethod
    def remove_one(cls, name: str, client: QdrantClient) -> bool:
        """
        Remove a repository and its indexed data.

        Deletes the repository's vector store collection and removes it from
        the configuration. This operation is permanent and cannot be undone.

        Args:
            name: Name of the repository to remove.
            client: QdrantClient instance for deleting the collection.

        Returns:
            True if the repository was successfully removed, False if it was
            already removed by another process (race condition).

        Raises:
            RepoNotFoundError: If no repository with the given name exists.
        """
        repo = cls.get_one(name)

        repo.store.delete_collection(client)

        # Delete cache directory
        cache_dir = repo.cache.cache_dir
        if cache_dir.exists():
            for file in cache_dir.iterdir():
                file.unlink()
            cache_dir.rmdir()

        # Remove from repos.json
        repo_settings = RepoSettings.load()
        new_repo_settings = [r for r in repo_settings if r.name != name]
        RepoSettings.save(new_repo_settings)
        if new_repo_settings != repo_settings:
            logger.info(f"Removed repository: {name}")
            return True
        return False

    @classmethod
    def remove_all(cls, client: QdrantClient) -> bool:
        """
        Remove all configured repositories and their indexed data.

        Deletes all repositories' vector store collections and clears the
        configuration. This operation is permanent and cannot be undone.

        Args:
            client: QdrantClient instance for deleting collections.

        Returns:
            True if any repositories were removed, False if there were none.
        """
        repos = cls.get_all()
        if not repos:
            return False

        for repo in repos:
            logger.info(f"Removing repository: {repo.name}")

            # Remove collection from store
            repo.store.delete_collection(client)

            # Delete cache directory
            cache_dir = repo.cache.cache_dir
            if cache_dir.exists():
                for file in cache_dir.iterdir():
                    file.unlink()
                cache_dir.rmdir()
                logger.info(f"Removed repository cache: {cache_dir}")

        # Clear repos.json
        RepoSettings.save([])
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

    def index(self, client: QdrantClient, full: bool = False) -> IndexResult:
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
            client: QdrantClient instance for vector store operations.
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

        # Initialize result
        result = IndexResult(repo=self.name, repo_path=self.path)

        # Ensure collection exists
        self.store.ensure_collection(client)

        # On full index, recreate the collection and clear cache
        if full:
            self.store.delete_collection(client)
            self._delete_hashmap()
            logger.info(f"Performing full index for repository: {self.name}")

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
            nodes_deleted = self.store.delete_nodes(client, hashes_to_delete)
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
                    self.store.upsert_nodes(client, nodes_to_upsert)
                    result.nodes_added += len(nodes_to_upsert)
                    nodes_to_upsert = []

            # Upsert any remaining nodes
            if nodes_to_upsert:
                self.store.upsert_nodes(client, nodes_to_upsert)
                result.nodes_added += len(nodes_to_upsert)
                nodes_to_upsert = []

        # Finalize result
        end_time = datetime.now(UTC)
        result.indexed_at = end_time
        result.duration = (end_time - start_time).total_seconds()

        # Save the new hashmap to cache
        self._set_hashmap(hashmap)

        return result

    def search(
        self,
        client: QdrantClient,
        query: str,
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
            client: QdrantClient instance for vector store operations.
            query: Natural language or code search query.
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
        search_results = self.store.search(
            client,
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
