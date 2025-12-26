"""Qdrant vector store integration with fastembed."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient, models

from indexter.config import settings
from indexter.config.store import StoreMode

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
                store_path = settings.store.path or (settings.data_dir / "store")
                store_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using local Qdrant storage at {store_path}")
                self._client = AsyncQdrantClient(path=str(store_path))
            elif mode == StoreMode.memory:
                # In-memory storage (for testing)
                logger.info("Using in-memory Qdrant storage")
                self._client = AsyncQdrantClient(location=":memory:")
            else:
                # Remote Qdrant server
                logger.info(f"Connecting to Qdrant (async) at {settings.store.url}")
                self._client = AsyncQdrantClient(
                    url=settings.store.url,
                    api_key=settings.store.api_key,
                    prefer_grpc=settings.store.use_grpc,
                )

            # Set the embedding model for fastembed
            self._client.set_model(settings.embedding.model_name)
            self._embedding_model_name = settings.embedding.model_name
            # Get the vector name used by fastembed (e.g., 'fast-bge-small-en-v1.5')
            if self._vector_name is None:
                vector_params = self._client.get_fastembed_vector_params()
                self._vector_name = list(vector_params.keys())[0]
            logger.info(
                f"Using embedding model (async): {settings.embedding.model_name} "
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


# Global store instance
store = VectorStore()
