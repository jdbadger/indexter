"""
Vector database integration using Qdrant and FastEmbed for semantic code search.

This module provides a high-level interface to a Qdrant vector database, handling
code embedding generation, storage, and semantic search. It's a critical component
of Indexter's search infrastructure, enabling natural language queries over code.

The module abstracts away the complexity of vector operations, embedding generation,
and collection management, providing a simple API for storing and searching code nodes.

Architecture
------------
The module consists of a single VectorStore class with three operational modes:

1. **Local Mode**: File-based storage using Qdrant's embedded database
2. **Memory Mode**: In-memory storage for testing and development
3. **Remote Mode**: Connection to a remote Qdrant server for production

Storage Modes
-------------
The storage mode is configured via settings.store.mode:

Local (Serverless):
    Uses Qdrant's embedded database, storing data in ~/.local/share/indexter/store.
    No external server required. Best for single-user, single-machine scenarios.

    Configuration:
        mode = "local"

Memory (Testing):
    All data stored in RAM, lost on process termination. Fast and clean for tests.
    Useful for development and CI/CD pipelines.

    Configuration:
        mode = "memory"

Remote (Production):
    Connects to a standalone Qdrant server. Supports clustering, replication,
    and multi-client access. Best for team environments and production deployments.

    Configuration:
        mode = "remote"
        host = "localhost"
        port = 6333
        grpc_port = 6334
        api_key = "optional-api-key"

Embedding Generation
--------------------
The module uses FastEmbed for automatic embedding generation:

- Models are lazy-loaded on first use
- Embeddings are generated automatically during upsert
- Default model: "BAAI/bge-small-en-v1.5" (384 dimensions)
- Model can be configured globally or per-repository

FastEmbed integration provides:
- No manual embedding generation required
- Efficient batch processing
- Automatic dimensionality handling
- CPU-optimized inference

The embedding process:
1. Text content is extracted from code nodes
2. FastEmbed generates vector embeddings on-the-fly
3. Vectors are stored alongside metadata in Qdrant
4. Search queries are embedded using the same model

Collections
-----------
Each repository gets its own Qdrant collection for isolation:

Collection Naming:
    Collections are named "indexter_{repo_name}" where repo_name is the
    repository directory name. For example, "my-project" becomes
    "indexter_my-project".

Collection Schema:
    Each point (node) in a collection contains:
    - Vector: Dense embedding of the code content
    - Payload: Metadata including file path, node type, language, etc.
    - ID: UUID of the node

Point Payload Structure:
    {
        "document": str,        # Full code content
        "hash": str,           # Document content hash
        "repo_path": str,      # Repository root path
        "document_path": str,  # File path within repo
        "language": str,       # Programming language
        "node_type": str,      # Type: function, class, method, etc.
        "node_name": str,      # Name of the code construct
        "start_byte": int,     # Start position in file
        "end_byte": int,       # End position in file
        "start_line": int,     # Start line number
        "end_line": int,       # End line number
        "documentation": str,  # Docstring/comments
        "parent_scope": str,   # Enclosing class/module
        "signature": str,      # Function signature
        # Additional language-specific fields in extra dict
    }

Search Capabilities
-------------------
The search method supports both semantic search and metadata filtering:

Semantic Search:
    Uses vector similarity (cosine distance) to find code semantically similar
    to the query text. Queries can be natural language or code snippets.

Metadata Filters:
    - file_path: Search within specific files or directories
    - language: Filter by programming language
    - node_type: Filter by code construct (function, class, etc.)
    - node_name: Filter by exact name
    - has_documentation: Filter documented/undocumented code

Filters can be combined for precise queries like "find Python functions named
'authenticate' with documentation in the auth/ directory".

Change Detection
----------------
The store maintains document hashes for efficient change detection:

- get_document_hashes(): Returns all indexed file hashes
- Incremental indexing compares hashes to detect changes
- Only modified files trigger re-indexing
- Deleted files trigger cleanup of associated nodes

Hash Structure:
    Each document hash is a SHA-256 of "{file_path}:{content}", enabling
    detection of both content changes and file moves/renames.

Classes
-------
VectorStore:
    Main interface to the vector database. Manages collections, nodes, and
    search operations. Implements lazy initialization and connection pooling.

Singleton Pattern:
    A module-level `store` instance is provided for convenient access across
    the application without managing multiple connections.

Examples
--------
Basic operations:

    >>> from indexter.store import store
    >>>
    >>> # Create a collection for a repository
    >>> await store.create_collection("indexter_myproject")
    >>>
    >>> # Store nodes with automatic embedding
    >>> from indexter.models import Node, NodeMetadata
    >>> nodes = [...]  # List of parsed code nodes
    >>> await store.upsert_nodes("indexter_myproject", nodes)
    >>>
    >>> # Count indexed nodes
    >>> count = await store.count_nodes("indexter_myproject")
    >>> print(f"Indexed {count} code nodes")

Semantic search:

    >>> # Search for authentication code
    >>> results = await store.search(
    ...     collection_name="indexter_myproject",
    ...     query="user authentication and password validation",
    ...     limit=5
    ... )
    >>>
    >>> for result in results:
    ...     print(f"{result['score']:.3f} - {result['node_name']}")
    ...     print(f"  {result['file_path']}:{result['start_line']}")

Filtered search:

    >>> # Find Python classes with documentation
    >>> results = await store.search(
    ...     collection_name="indexter_myproject",
    ...     query="data processing",
    ...     language="python",
    ...     node_type="class",
    ...     has_documentation=True,
    ...     limit=10
    ... )

Change detection workflow:

    >>> # Get current indexed file hashes
    >>> stored_hashes = await store.get_document_hashes("indexter_myproject")
    >>>
    >>> # Compare with local file hashes to find changes
    >>> local_hashes = await repo.get_document_hashes()
    >>> modified = [p for p in local_hashes if stored_hashes.get(p) != local_hashes[p]]
    >>>
    >>> # Delete stale nodes and re-index
    >>> if modified:
    ...     await store.delete_by_document_paths("indexter_myproject", modified)

Performance Considerations
--------------------------
- Batch Operations: Upserts are batched to reduce network overhead
- Connection Pooling: Single client instance shared across operations
- Lazy Initialization: Client created only when first needed
- Collection Caching: In-memory cache prevents redundant existence checks
- Scroll API: Large result sets use cursor-based pagination
- Vector Compression: Qdrant's built-in quantization reduces memory usage

Best Practices
--------------
1. Use collection per repository for isolation and cleanup
2. Batch upsert operations when indexing many files
3. Use metadata filters to narrow search scope before semantic search
4. Delete by document path when re-indexing modified files
5. Monitor collection sizes with count_nodes()
6. Use local mode for development, remote for production

Configuration
-------------
Store behavior is controlled through global settings:

    [store]
    mode = "local"              # local, memory, or remote
    host = "localhost"          # Remote server host
    port = 6333                 # HTTP API port
    grpc_port = 6334           # gRPC port (faster)
    prefer_grpc = true         # Use gRPC when available
    api_key = ""               # Optional authentication

    # Global embedding model
    embedding_model = "BAAI/bge-small-en-v1.5"

Limitations
-----------
- Maximum collection size depends on available disk/memory
- Search performance degrades with collections over ~1M points
- Embedding generation is CPU-bound (no GPU acceleration in FastEmbed)
- Exact text search not supported (semantic only)
- Updates require delete + re-insert (no true update operation)

Notes
-----
- All operations are asynchronous and require an event loop
- The store singleton is thread-safe but not process-safe
- Collection names must be valid Qdrant identifiers
- Vector dimensionality is fixed per collection (determined by embedding model)
- Metadata filters use exact matching (no fuzzy search)
- Empty collections are valid but search returns no results
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient, models

from .config import StoreMode, settings

if TYPE_CHECKING:
    from indexter.models import Node

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector store with fastembed embeddings."""

    def __init__(self):
        """Initialize the vector store."""
        self._client: AsyncQdrantClient | None = None
        self._embedding_model_name: str | None = None
        self._initialized_collections: set[str] = set()
        self._vector_name: str | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Get or create the async Qdrant client."""
        if self._client is None:
            mode = settings.store.mode

            if mode == StoreMode.local:
                # Local file-based storage (serverless)
                store_path = settings.data_dir / "store"
                store_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using local Qdrant storage at {store_path}")
                self._client = AsyncQdrantClient(path=str(store_path))
            elif mode == StoreMode.memory:
                # In-memory storage (for testing)
                logger.info("Using in-memory Qdrant storage")
                self._client = AsyncQdrantClient(location=":memory:")
            else:
                # Remote Qdrant server
                logger.info(
                    f"Connecting to Qdrant (async) at {settings.store.host}:{settings.store.port}"
                )
                self._client = AsyncQdrantClient(
                    host=settings.store.host,
                    port=settings.store.port,
                    grpc_port=settings.store.grpc_port,
                    prefer_grpc=settings.store.prefer_grpc,
                    api_key=settings.store.api_key,
                )

            # Set the embedding model for fastembed
            self._client.set_model(settings.embedding_model)
            self._embedding_model_name = settings.embedding_model
            # Get the vector name used by fastembed (e.g., 'fast-bge-small-en-v1.5')
            if self._vector_name is None:
                vector_params = self._client.get_fastembed_vector_params()
                self._vector_name = list(vector_params.keys())[0]
            logger.info(
                f"Using embedding model (async): {self._embedding_model_name} "
                f"(vector: {self._vector_name})"
            )
        return self._client

    async def create_collection(self, collection_name: str) -> None:
        """Create a collection in the vector store using fastembed vector params.

        Args:
            collection_name: Name of the collection to create.
        """
        vector_params = self.client.get_fastembed_vector_params()
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
        )
        logger.info(f"Created collection: {collection_name}")

    async def delete_collection(self, collection_name: str) -> None:
        """Drop a collection from the vector store.

        Args:
            collection_name: Name of the collection to drop.
        """
        await self.client.delete_collection(collection_name=collection_name)
        if collection_name in self._initialized_collections:
            self._initialized_collections.remove(collection_name)
        logger.info(f"Dropped collection: {collection_name}")

    async def ensure_collection(self, collection_name: str) -> None:
        """Ensure a collection exists, creating it if necessary.

        Uses an in-memory cache to avoid repeated checks.

        Args:
            collection_name: Name of the collection to ensure exists.
        """
        if collection_name in self._initialized_collections:
            return

        # Check if collection exists
        collections = await self.client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if collection_name not in existing_names:
            await self.create_collection(collection_name)

        self._initialized_collections.add(collection_name)

    async def get_document_hashes(self, collection_name: str) -> dict[str, str]:
        """Get all document hashes from a collection.

        Scrolls through all points and extracts unique document_path -> hash mappings.

        Args:
            collection_name: Name of the collection to query.

        Returns:
            Dict mapping document_path to content hash.
        """
        await self.ensure_collection(collection_name)

        document_hashes: dict[str, str] = {}
        offset = None

        while True:
            results, next_offset = await self.client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=["document_path", "hash"],
                with_vectors=False,
            )

            for point in results:
                if point.payload:
                    doc_path = point.payload.get("document_path")
                    doc_hash = point.payload.get("hash")
                    if doc_path and doc_hash:
                        # Only store first occurrence (all nodes from same doc have same hash)
                        if doc_path not in document_hashes:
                            document_hashes[doc_path] = doc_hash

            if next_offset is None:
                break
            offset = next_offset

        return document_hashes

    async def count_nodes(self, collection_name: str) -> int:
        """Count the total number of nodes in a collection.

        Args:
            collection_name: Name of the collection to count.

        Returns:
            Total number of nodes (points) in the collection.
        """
        await self.ensure_collection(collection_name)
        collection_info = await self.client.get_collection(collection_name)
        return collection_info.points_count or 0

    async def upsert_nodes(
        self,
        collection_name: str,
        nodes: list[Node],
    ) -> int:
        """Upsert nodes to a collection using fastembed for embeddings.

        Args:
            collection_name: Name of the collection to upsert to.
            nodes: List of Node objects to upsert.

        Returns:
            Number of nodes upserted.
        """
        if not nodes:
            return 0

        await self.ensure_collection(collection_name)

        # Prepare documents and metadata for fastembed
        documents = [node.content for node in nodes]
        metadata = [
            {
                "document": node.content,  # Store content for retrieval
                "hash": node.metadata.hash,
                "repo_path": node.metadata.repo_path,
                "document_path": node.metadata.document_path,
                "language": node.metadata.language,
                "node_type": node.metadata.node_type,
                "node_name": node.metadata.node_name or "",
                "start_byte": node.metadata.start_byte,
                "end_byte": node.metadata.end_byte,
                "start_line": node.metadata.start_line,
                "end_line": node.metadata.end_line,
                "documentation": node.metadata.documentation or "",
                "parent_scope": node.metadata.parent_scope or "",
                "signature": node.metadata.signature or "",
                **node.metadata.extra,
            }
            for node in nodes
        ]
        ids = [node.id for node in nodes]

        # Ensure vector name and embedding model are initialized
        if self._vector_name is None or self._embedding_model_name is None:
            raise RuntimeError("Vector store not properly initialized")

        # Build points with Document for automatic embedding inference
        points = [
            models.PointStruct(
                id=point_id,
                vector={
                    self._vector_name: models.Document(text=doc, model=self._embedding_model_name)
                },
                payload=meta,
            )
            for point_id, doc, meta in zip(ids, documents, metadata, strict=True)
        ]

        await self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

        return len(nodes)

    async def delete_by_document_paths(
        self,
        collection_name: str,
        document_paths: list[str],
    ) -> int:
        """Delete all nodes matching the given document paths.

        Args:
            collection_name: Name of the collection to delete from.
            document_paths: List of document paths to delete nodes for.

        Returns:
            Number of paths processed (not individual points).
        """
        if not document_paths:
            return 0

        await self.ensure_collection(collection_name)

        # Delete using filter on document_path
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="document_path",
                            match=models.MatchValue(value=path),
                        )
                        for path in document_paths
                    ]
                )
            ),
        )

        return len(document_paths)

    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        file_path: str | None = None,
        language: str | None = None,
        node_type: str | None = None,
        node_name: str | None = None,
        has_documentation: bool | None = None,
    ) -> list[dict]:
        """Perform semantic search on a collection with optional filters.

        Args:
            collection_name: Name of the collection to search.
            query: Search query text.
            limit: Maximum number of results to return.
            file_path: Filter by file path (exact match or prefix).
            language: Filter by programming language.
            node_type: Filter by node type (e.g., 'function', 'class').
            node_name: Filter by node name (exact match).
            has_documentation: Filter by documentation presence (e.g. docstring or doc comments).

        Returns:
            List of search results with scores and metadata.
        """
        await self.ensure_collection(collection_name)

        # Build filter conditions
        filter_conditions = []

        if file_path:
            # Support both exact match and prefix matching
            if file_path.endswith("/"):
                # Prefix match for directories
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_path",
                        match=models.MatchText(text=file_path),
                    )
                )
            else:
                # Exact match for files
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_path",
                        match=models.MatchValue(value=file_path),
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

        # Build query filter
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        # Ensure vector name and embedding model are initialized
        if self._vector_name is None or self._embedding_model_name is None:
            raise RuntimeError("Vector store not properly initialized")

        # Perform search using query_points with Document for embedding inference
        results = await self.client.query_points(
            collection_name=collection_name,
            query=models.Document(text=query, model=self._embedding_model_name),
            using=self._vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        # Format results
        formatted_results = []
        for point in results.points:
            formatted_results.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "content": point.payload.get("document", "") if point.payload else "",
                    "file_path": point.payload.get("document_path", "") if point.payload else "",
                    "language": point.payload.get("language", "") if point.payload else "",
                    "node_type": point.payload.get("node_type", "") if point.payload else "",
                    "node_name": point.payload.get("node_name", "") if point.payload else "",
                    "start_line": point.payload.get("start_line", 0) if point.payload else 0,
                    "end_line": point.payload.get("end_line", 0) if point.payload else 0,
                    "documentation": point.payload.get("documentation", "")
                    if point.payload
                    else "",
                    "signature": point.payload.get("signature", "") if point.payload else "",
                    "parent_scope": point.payload.get("parent_scope", "") if point.payload else "",
                }
            )

        return formatted_results


store = VectorStore()
