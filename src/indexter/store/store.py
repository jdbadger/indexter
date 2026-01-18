"""
Qdrant vector store integration for semantic code search.

This module provides the vector database layer for Indexter, handling storage
and retrieval of code embeddings using Qdrant. It supports hybrid search
combining dense and sparse vectors with Reciprocal Rank Fusion (RRF).

The ``VectorStore`` class is the primary interface for all vector operations,
including collection management, node storage, and semantic search. It should
be used as an async context manager to ensure proper resource cleanup.

Architecture:
    The store uses Qdrant's async client with FastEmbed for embedding generation.
    Embeddings are created on-the-fly during upsert and search operations,
    eliminating the need for a separate embedding service.

    Storage modes:

    - **remote**: Connect to Qdrant server running in Docker (default)
    - **memory**: In-memory storage for testing

    The Qdrant container is managed via the CLI: ``indexter store init``.
    Data is persisted at ``~/.local/share/indexter/qdrant`` (XDG_DATA_HOME).

    Hybrid search:
        Search combines two embedding strategies using RRF fusion:

        1. Dense vectors (sentence-transformers) for semantic similarity
        2. Sparse vectors (BM25) for keyword matching

        This hybrid approach provides better results than either method alone,
        especially for code search where exact identifiers matter.

Collections:
    Each repository has its own collection named ``indexter_{repo_name}``.
    Collections are created lazily on first access and cached to avoid
    repeated existence checks.

    Collection schema includes:

    - Dense vector field for semantic embeddings
    - Sparse vector field for BM25 embeddings
    - Payload fields: content, document_path, hash, language, node_type,
      node_name, parent_scope, documentation, signature, etc.

Classes:
    VectorStore:
        Async Qdrant client wrapper with methods for collection management,
        node upsert, deletion, and hybrid semantic search.

    SearchResult (in models.py):
        Single search result with content, score, and metadata.

    SearchResults (in models.py):
        Container for search results with query and filter information.

    IndexResult (in models.py):
        Statistics from an indexing operation (nodes added, updated, etc.).

Example:
    Using VectorStore as a context manager::

        from indexter.store import VectorStore

        async with VectorStore() as store:
            # Ensure collection exists
            await store.ensure_collection("indexter_my-project")

            # Upsert nodes (embeddings generated automatically)
            nodes = [Node(content="def hello(): pass", metadata=...)]
            count = await store.upsert_nodes("indexter_my-project", nodes)

            # Search with filters
            results = await store.search(
                collection_name="indexter_my-project",
                query="function that handles authentication",
                language="python",
                node_type="function",
                limit=10,
            )

            for result in results.results:
                print(f"{result.score:.3f}: {result.metadata['node_name']}")

    Change detection for incremental indexing::

        async with VectorStore() as store:
            # Get stored hashes for comparison with local files
            hashes = await store.get_document_hashes("indexter_my-project")
            # Returns: {"src/main.py": "abc123...", "src/utils.py": "def456..."}

Configuration:
    Store behavior is controlled via ``settings.store``:

    - ``image``: Docker image for Qdrant (default: qdrant/qdrant:latest)
    - ``host``, ``port``: Connection settings (default: localhost:6333)
    - ``grpc_port``: gRPC port (default: 6334)
    - ``prefer_grpc``: Whether to use gRPC over HTTP
    - ``api_key``: Authentication for cloud instances
    - ``mode``: Storage mode (server or memory for testing)
    - ``timeout``: API operation timeout in seconds (default: 120)

    When ``prefer_grpc`` is enabled, the client is configured with keepalive
    settings to detect stale connections and automatic retry policies to
    recover from transient failures (e.g., server restarts).

    Embedding models are configured globally:

    - ``settings.embedding_model``: Dense embedding model
    - ``settings.sparse_embedding_model``: Sparse/BM25 model

Note:
    - All operations are asynchronous
    - The client is lazily initialized on first access
    - Collection existence is cached to reduce API calls
    - Nodes are processed in batches to manage memory during embedding

See Also:
    - ``indexter.config.StoreSettings``: Store configuration options
    - ``indexter.parser.models.Node``: Node model stored in collections
    - ``indexter.models.Repo.search``: High-level search interface
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient, models

from indexter.config import StoreMode, settings

from .models import SearchResult, SearchResults

if TYPE_CHECKING:
    from indexter.parser.models import Node

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector store with fastembed embeddings."""

    def __init__(self):
        """Initialize the vector store."""
        self._client: AsyncQdrantClient | None = None
        self._embedding_model_name: str | None = None
        self._sparse_embedding_model_name: str | None = None
        self._initialized_collections: set[str] = set()
        self._vector_name: str | None = None
        self._sparse_vector_name: str | None = None

    async def __aenter__(self) -> VectorStore:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and close client connection."""
        await self.close()

    async def close(self) -> None:
        """Close the client connection and reset state.

        This should be called when the event loop is about to change (e.g.,
        between CLI commands that each use their own anyio.run() call) to
        prevent gRPC connection issues.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._embedding_model_name = None
            self._sparse_embedding_model_name = None
            self._initialized_collections = set()
            self._vector_name = None
            self._sparse_vector_name = None
            logger.debug("Closed Qdrant client connection")

    @property
    def client(self) -> AsyncQdrantClient:
        """Get or create the async Qdrant client."""
        if self._client is None:
            mode = settings.store.mode

            if mode == StoreMode.memory:
                # In-memory storage (for testing)
                logger.info("Using in-memory Qdrant storage")
                self._client = AsyncQdrantClient(
                    location=":memory:",
                    timeout=settings.store.timeout,
                )
            else:
                # Qdrant server (default)
                logger.info(f"Connecting to Qdrant (async) at {settings.store.host}:{settings.store.port}")

                if settings.store.prefer_grpc:
                    # gRPC retry policy to handle transient failures
                    grpc_service_config = json.dumps(
                        {
                            "retryPolicy": {
                                "maxAttempts": 3,
                                "initialBackoff": "0.1s",
                                "maxBackoff": "1s",
                                "backoffMultiplier": 2,
                                "retryableStatusCodes": ["UNAVAILABLE"],
                            }
                        }
                    )
                    # gRPC keepalive options to detect stale connections and trigger reconnection
                    # This helps recover from server restarts without "Socket Closed" errors
                    grpc_options = {
                        "grpc.keepalive_time_ms": 10000,  # Send keepalive ping every 10 seconds
                        "grpc.keepalive_timeout_ms": 5000,  # Wait 5 seconds for ping ack
                        "grpc.keepalive_permit_without_calls": True,  # Send pings even when idle
                        "grpc.http2.max_pings_without_data": 0,  # Allow unlimited pings
                        "grpc.enable_retries": 1,  # Enable automatic retries
                        "grpc.service_config": grpc_service_config,
                    }
                else:
                    grpc_options = None

                self._client = AsyncQdrantClient(
                    host=settings.store.host,
                    port=settings.store.port,
                    grpc_port=settings.store.grpc_port,
                    prefer_grpc=settings.store.prefer_grpc,
                    api_key=settings.store.api_key,
                    timeout=settings.store.timeout,
                    grpc_options=grpc_options,
                )

            # Set the embedding models for fastembed
            self._client.set_model(settings.embedding_model)
            self._embedding_model_name = settings.embedding_model
            self._client.set_sparse_model(settings.sparse_embedding_model)
            self._sparse_embedding_model_name = settings.sparse_embedding_model
            # Get the vector names used by fastembed (e.g., 'fast-bge-small-en-v1.5')
            if self._vector_name is None:
                if vector_params := self._client.get_fastembed_vector_params():
                    self._vector_name = list(vector_params.keys())[0]
            if self._sparse_vector_name is None:
                if sparse_vector_params := self._client.get_fastembed_sparse_vector_params():
                    self._sparse_vector_name = list(sparse_vector_params.keys())[0]
            logger.info(
                f"Using embedding model (async): {self._embedding_model_name} "
                f"(vector: {self._vector_name}, sparse vector: {self._sparse_vector_name})"
            )
        return self._client

    async def create_collection(self, collection_name: str) -> None:
        """Create a collection in the vector store using fastembed vector params.

        Args:
            collection_name: Name of the collection to create.
        """
        vector_params = self.client.get_fastembed_vector_params()
        sparse_vector_params = self.client.get_fastembed_sparse_vector_params()
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
            sparse_vectors_config=sparse_vector_params,
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
        batch_size: int = 10,
    ) -> int:
        """Upsert nodes to a collection using fastembed for embeddings.

        Processes nodes in small sub-batches to reduce memory pressure during
        embedding generation. This is important because FastEmbed loads models
        into memory and generates embeddings for all texts at once.

        Args:
            collection_name: Name of the collection to upsert to.
            nodes: List of Node objects to upsert.
            batch_size: Number of nodes to process in each sub-batch. Smaller
                values reduce memory usage but increase API calls. Default is 10.

        Returns:
            Number of nodes upserted.
        """
        if not nodes:
            return 0

        await self.ensure_collection(collection_name)

        vector_name = self._vector_name
        embedding_model_name = self._embedding_model_name
        sparse_vector_name = self._sparse_vector_name
        sparse_embedding_model_name = self._sparse_embedding_model_name

        if (
            vector_name is None
            or embedding_model_name is None
            or sparse_vector_name is None
            or sparse_embedding_model_name is None
        ):
            raise RuntimeError("Vector store not properly initialized")

        total_upserted = 0

        # Process nodes in small sub-batches to reduce memory pressure
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i : i + batch_size]

            # Prepare documents and metadata for this batch only
            texts = [node.content for node in batch_nodes]
            payloads = [node.as_payload() for node in batch_nodes]
            ids = [node.id for node in batch_nodes]

            # Build points with Document for automatic embedding inference
            points = [
                models.PointStruct(
                    id=id,
                    vector={
                        vector_name: models.Document(text=text, model=embedding_model_name),
                        sparse_vector_name: models.Document(text=text, model=sparse_embedding_model_name),
                    },
                    payload=payload,
                )
                for id, text, payload in zip(ids, texts, payloads, strict=True)
            ]

            await self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_upserted += len(batch_nodes)

            # Clear references to allow garbage collection
            del points, texts, payloads, ids, batch_nodes

        return total_upserted

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

        Returns:
            List of SearchResult objects.
        """
        await self.ensure_collection(collection_name)

        # Build filter conditions
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

        # Build query filter
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        # Ensure vector name and embedding model are initialized
        vector_name = self._vector_name
        embedding_model_name = self._embedding_model_name
        sparse_vector_name = self._sparse_vector_name
        sparse_embedding_model_name = self._sparse_embedding_model_name

        if (
            vector_name is None
            or embedding_model_name is None
            or sparse_vector_name is None
            or sparse_embedding_model_name is None
        ):
            raise RuntimeError("Vector store not properly initialized")

        # Perform search using query_points with Document for embedding inference
        results = await self.client.query_points(
            collection_name=collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # we are using reciprocal rank fusion here
            ),
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=embedding_model_name),
                    using=vector_name,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=sparse_embedding_model_name),
                    using=sparse_vector_name,
                ),
            ],
            query_filter=query_filter,
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
